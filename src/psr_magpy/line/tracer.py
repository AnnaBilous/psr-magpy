import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from astropy.coordinates.representation import CartesianRepresentation

from psr_magpy.line.field_line import FieldLine, PropertyName
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)

class FieldLineTracer:
    def __init__(self, field: Any, R_max: float = 10.0, min_R_max_line: float = 1.0, dt: float = 0.01) -> None:
        """Initialize the field line tracer.

        Parameters
        ----------
        field : MultipoleField
            Object that calculates the magnetic field
        R_max : float
            Maximum radius to trace field lines to (in NS radii)
        min_R_max_line : float
            Minimum value for R_max to save the line
        dt : float
            Integration time step
        """
        self.field = field
        self.R_max = R_max
        self.min_R_max_line = min_R_max_line
        self.dt = dt
        
        logger.info(f"Initialized FieldLineTracer with R_max={R_max}, min_R_max_line={min_R_max_line}, dt={dt}")
        logger.debug(f"Field object: {field}")

    def hit_surface(self, t: float, r: np.ndarray, K: float) -> float:
        r = np.sum(r**2) ** 0.5
        return r - 0.999999

    hit_surface.terminal = True

    def hit_Rmax(self, t: float, r: np.ndarray, K: float) -> float:
        r = np.sum(r**2) ** 0.5
        return r - self.R_max

    hit_Rmax.terminal = True

    #def hit_Rmin(self, t: float, r: np.ndarray, K: float) -> float:
    #    r = np.sum(r**2) ** 0.5
   #     return r - self.R_min

    #hit_Rmin.terminal = True

    def _field_for_integration(self, t: float, r: np.ndarray, K: float) -> np.ndarray:
        """Wrapper for field calculation that includes unused time parameter required by solve_ivp.

        Parameters
        ----------
        t : float
            Time parameter (unused, required by solve_ivp)
        r : ndarray
            Position vector
        K : float
            Field direction multiplier

        Returns
        -------
        ndarray
            Magnetic field vector
        """
        return self.field.get_total_field(r, K, normalize=True)

    def trace_single_line(
        self,
        r_initial: CartesianRepresentation,
        footpoint_id: int,
        template: FieldLine,
        save_dR: bool = False,
    ) -> FieldLine:
        """Trace a single field line from the given initial position.

        Parameters
        ----------
        r_initial : CartesianRepresentation
            Initial position for field line tracing
        template : FieldLine
            Template specifying which properties to save
        dt : float
            Integration time step
        save_dR : bool
            Whether to save the median step size

        Returns
        -------
        FieldLine
            FieldLine object containing requested variables for the field line
        """
        output = template.copy_structure()
        output.set_property(footpoint_id=footpoint_id)
        r_initial = r_initial.xyz.value
        B_initial = self.field.get_total_field(r_initial, K=1, normalize=False)
        B_initial_norm = self.field.get_total_field(r_initial, K=1, normalize=True)
        dotprod = np.dot(B_initial_norm, r_initial)
        K = np.sign(dotprod)
        if K == 0:
            K = 1.0
        # Include hit_Rmin event to ensure we don't trace too far down?
        field_line = solve_ivp(
            self._field_for_integration,
            [0, 1e8],
            r_initial,
            args=[K],
            dense_output=True,
            events=(self.hit_surface, self.hit_Rmax),
            rtol=5.0e-8,
            atol=5.0e-8,
        )

        if isinstance(self.dt, int):
            t = np.linspace(field_line.t[0], field_line.t[-1] + self.dt, self.dt)
        else:
            t = np.arange(field_line.t[0], field_line.t[-1] + self.dt, self.dt)

        r = field_line.sol(t)
       
        props = self.calculate_line_properties(footpoint_id, r, B_initial, dotprod, K, template)
      
        output.set_property(**props)

        if save_dR:
            dr = r[:, 1:] - r[:, :-1]
            dR = np.median(np.sum(dr**2, axis=0) ** 0.5)
            output.dR = dR
        return output
    
    def calculate_line_properties(self, footpoint_id: int, r: np.ndarray, 
                                 B_initial: np.ndarray, dotprod: float, 
                                 K: float, template: FieldLine) -> Dict[str, Any]:
        """Calculate properties for a field line.
        
        Parameters
        ----------
        footpoint_id : int
            ID of the footpoint
        r : np.ndarray
            Position vectors along the field line
        B_initial : np.ndarray
            Initial magnetic field vector
        dotprod : float
            Dot product of normalized initial B field and position
        K : float
            Field direction multiplier
        template : FieldLine
            Template specifying which properties to save
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of calculated properties
        """
        line_R_max = np.max(np.sum(r**2, axis=0) ** 0.5)
        props: Dict[str, Any] = {}
        for prop in PropertyName:
            if prop.key_name == "footpoint_id":
                props[prop.key_name] = footpoint_id
            elif prop.key_name == "r_line":
                props[prop.key_name] = self.prep_r_line(r, line_R_max, template)
            elif prop.key_name == "r_max":
                props[prop.key_name] = line_R_max if template.save_flags.save_r_max else None
            elif prop.key_name == "r_cyl_max":
                props[prop.key_name] = (
                    np.max(np.sum(r[:2,] ** 2, axis=0) ** 0.5)
                    if template.save_flags.save_r_cyl_max
                    else None
                )
            elif prop.key_name == "z_abs_max":
                props[prop.key_name] = (
                    np.max(np.abs(r[2, :])) if template.save_flags.save_z_abs_max else None
                )
            elif prop.key_name == "b_ini":
                props[prop.key_name] = (
                    np.sum(B_initial**2) ** 0.5 if template.save_flags.save_b_ini else None
                )
            elif prop.key_name == "b_r_ini_angle":
                props[prop.key_name] = (
                    np.arccos(dotprod) if template.save_flags.save_b_r_ini_angle else None
                )
            elif prop.key_name == "curvature_line":
                props[prop.key_name] = self.prep_curvature_line(r, line_R_max, template)
            elif prop.key_name == "pa_line":
                props[prop.key_name] = self.prep_pa_line(r, K, line_R_max, template)
                
        # Remove None values from kwargs
        props = {k: v for k, v in props.items() if v is not None}
        return props
            
    def prep_r_line(self, r: np.ndarray, line_R_max: float, 
                   template: FieldLine) -> Optional[np.ndarray]:
        """Prepare position vectors for saving.
        
        Parameters
        ----------
        r : np.ndarray
            Position vectors along the field line
        line_R_max : float
            Maximum radius of the field line
        template : FieldLine
            Template specifying which properties to save
            
        Returns
        -------
        Optional[np.ndarray]
            Position vectors to save, or None if not saving
        """
        if template.save_flags.save_r_line:
            if line_R_max >= self.min_R_max_line:
                r_line_save = r.T
            else:                
                r_line_save = np.array([])  # Line traced but didn't reach min_R_max_line
        else:
                r_line_save = None  # Don't save line at all
        return r_line_save
    
    def prep_curvature_line(self, r: np.ndarray, line_R_max: float, template: FieldLine) -> np.ndarray:
        """Calculate curvature along the field line.
        
        Parameters
        ----------
        r : np.ndarray
            Position vectors along the field line
            
        Returns
        -------
        np.ndarray
            Curvature values along the field line
        """
        if template.save_flags.save_curvature_line:
            if line_R_max >= self.min_R_max_line:
                dr = np.gradient(r, axis=0)     
                ddr = np.gradient(dr, axis=0)
                kappa1 = (np.linalg.norm(dr, axis=1))**2*(np.linalg.norm(ddr, axis=1))**2
                kappa2 = (np.sum(dr*ddr, axis=1))**2
                kappa3 = (np.linalg.norm(dr, axis=1))**3
                kappa = ((kappa1 - kappa2)**0.5/kappa3)**-1
            else:
                kappa = np.array([])
        else:
            kappa = None
        
        return kappa


    def prep_pa_line(self, r: np.ndarray, B_dir: float, line_R_max: float, template: FieldLine) -> np.ndarray:
        """Calculate position angle along the field line.
        
        Parameters
        ----------
        r : np.ndarray
            Position vectors along the field line
        B_dir : float
            Field direction multiplier
            
        Returns
        -------
        np.ndarray
            Position angle values along the field line
        """
        if template.save_flags.save_pa_line:
            if line_R_max >= self.min_R_max_line:
                Npoints = r.shape[1]
                dr = np.gradient(r, axis=1)
                dr_norm = dr/np.linalg.norm(dr, axis=0)[np.newaxis,:]
                n = np.gradient(dr_norm, axis=1)
                n_abs = np.linalg.norm(n, axis=0)
                n_abs[n_abs==0] = 1.0
                n_norm = n/n_abs[np.newaxis,:]
                dr_R = np.linalg.norm(dr, axis=0)
                dr_theta = np.arccos(dr[2,:]/dr_R)
                dr_theta[dr_R==0] = 0
                dr_rp = (dr[0,:]**2+dr[1,:]**2)**0.5
                dr_phi = np.arctan2(dr[1,:], dr[0,:])
                dr_phi[dr_phi<0] += 2*np.pi
                dr_phi[dr_rp==0] = 0
        
                l0 = np.zeros((3,Npoints), dtype=float)
                l0[0,:] = -np.sin(dr_phi)
                l0[1,:] = np.cos(dr_phi)

                m0 = np.zeros((3,Npoints), dtype=float)
                m0[0,:] = np.cos(dr_theta)*np.cos(dr_phi)
                m0[1,:] = np.cos(dr_theta)*np.sin(dr_phi)
                m0[2,:] = -np.sin(dr_theta)
                PA = np.arctan2( np.sum(n_norm*l0, axis=0), np.sum(n_norm*m0 , axis=0))

                if B_dir == -1:
                    PA = PA + np.pi
                PA[PA<-np.pi] += 2*np.pi
                PA[PA>np.pi] -= 2*np.pi
            else:
                PA = np.array([])
        else:
            PA = None
        return PA

    

