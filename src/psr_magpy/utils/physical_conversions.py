
def get_R_LC(P_ms: float):
    """Calculate the light cylinder radius (in R_NS) 
    for a given pulsar period in milliseconds."""
    return 4.77 * P_ms      

def get_P_ms(R_LC: float):
    """Calculate the pulsar period in milliseconds for a given light cylinder radius (in R_NS)."""
    return R_LC / 4.77