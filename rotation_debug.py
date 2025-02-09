import numpy as np
from scipy.spatial.transform import Rotation as rot
from astropy.coordinates import Angle
import astropy.units as u
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def debug_rotation(pole_lat_deg: float, vector: np.ndarray, plot: bool = True) -> None:
    """Debug rotation operation with given pole latitude and vector.
    
    Args:
        pole_lat_deg: Pole latitude in degrees
        vector: Input vector to rotate (shape: (3,) or (N, 3))
        plot: Whether to create 3D visualization
    """
    # Convert pole_lat to Angle for consistency
    pole_lat = Angle(pole_lat_deg * u.deg)
    
    # Create rotation operator
    rotation_operator = rot.from_euler("yzx", [pole_lat.rad, 0, 0])
    
    # Ensure vector is 2D array (N, 3)
    vector = np.atleast_2d(vector)
    
    # Apply rotation
    rotated_vector = rotation_operator.apply(vector)
    
    # Print results
    print(f"Input parameters:")
    print(f"  Pole latitude: {pole_lat_deg} degrees")
    print(f"  Input vector(s):\n{vector}")
    print(f"\nRotation matrix:\n{rotation_operator.as_matrix()}")
    print(f"\nRotated vector(s):\n{rotated_vector}")
    
    # Calculate vector magnitude before and after to verify preservation
    mag_before = np.linalg.norm(vector, axis=1)
    mag_after = np.linalg.norm(rotated_vector, axis=1)
    print(f"\nMagnitude check:")
    print(f"  Before: {mag_before}")
    print(f"  After:  {mag_after}")
    
    if plot:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot original vectors
        for i in range(len(vector)):
            ax.quiver(0, 0, 0, 
                     vector[i, 0], vector[i, 1], vector[i, 2],
                     color='b', alpha=0.5, label='Original' if i == 0 else None)
        
        # Plot rotated vectors
        for i in range(len(rotated_vector)):
            ax.quiver(0, 0, 0, 
                     rotated_vector[i, 0], rotated_vector[i, 1], rotated_vector[i, 2],
                     color='r', alpha=0.5, label='Rotated' if i == 0 else None)
        
        # Plot rotation axis (y-axis for first rotation)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', alpha=0.5, label='Rotation Axis (y)')
        
        # Set plot limits and labels
        max_range = np.max(np.abs(np.vstack((vector, rotated_vector))))
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Add title and legend
        ax.set_title(f'Vector Rotation (Pole Latitude: {pole_lat_deg}°)')
        ax.legend()
        
        # Add coordinate system
        ax.plot([-max_range, max_range], [0, 0], [0, 0], 'k--', alpha=0.2)
        ax.plot([0, 0], [-max_range, max_range], [0, 0], 'k--', alpha=0.2)
        ax.plot([0, 0], [0, 0], [-max_range, max_range], 'k--', alpha=0.2)
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Test with single vector along z-axis
    test_vector1 = np.array([0, 0, 1])
    print("Test 1: Rotating z-axis vector")
    print("-" * 50)
    debug_rotation(45, test_vector1)
    
    print("\nTest 2: Rotating multiple vectors")
    print("-" * 50)
    # Test with multiple vectors
    test_vectors = np.array([
        [1, 0, 0],  # x-axis
        [0, 1, 0],  # y-axis
        [0, 0, 1]   # z-axis
    ])
    debug_rotation(30, test_vectors)