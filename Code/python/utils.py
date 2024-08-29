import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np

magnetization = 1.27e6  # Magnetization of the magnet in A/m

def create_cylinder_magnet(radius, length, magnetization=magnetization):
    """
    Create a cylindrical magnet using magpylib.
    
    Parameters:
    - magnetization: Magnetization vector
    - radius: Radius of the cylinder
    - length: Length of the cylinder
    
    Returns:
    - Cylindrical magnet object
    """
    magnet = magpy.magnet.Cylinder(magnetization=(0, 0, magnetization), dimension=(radius * 2, length))
    return magnet


def create_optimized_multi_cylinder_magnet(radius, magnet_length, number_of_magnets, gap, magnetization, use_alternation=False):
    """
    Create an object consisting of multiple cylinder magnets. The magnets are aligned on the z-axis and have a gap between them.
    
    Parameters:
    - radius: Radius of the cylinder (m)
    - magnet_length: Length of each cylinder (m)
    - number_of_magnets: Number of cylinder magnets
    - gap: Gap between the cylinders (m)
    - magnetization: Magnetization vector (A/m)
    - use_alternation: If set to True, alternates the polarity of neighboring magnets
    
    Returns:
    - Multi-cylinder magnet object centered at the origin
    """
    cylinders = []
    
    for i in range(number_of_magnets):
        cylinders.append(magpy.magnet.Cylinder(
            magnetization=(0, 0, -magnetization if i % 2 == 1 and use_alternation else magnetization),
            dimension=(radius * 2, magnet_length),
        ))
    
    # Create a collection to hold the magnets
    collection = magpy.Collection(*cylinders)
    
    # Calculate the total length of the magnet system including gaps
    total_length = number_of_magnets * magnet_length + (number_of_magnets - 1) * gap
    
    # Calculate the starting position to center the system
    start_z = -total_length / 2 + magnet_length / 2
    
    # Position each magnet
    for i, cylinder in enumerate(cylinders):
        z_position = start_z + i * (magnet_length + gap)
        cylinder.move((0, 0, z_position))
    
    return collection

def plot_magnet(magnet_collection):
    """
    Plot the 3D view of the magnet collection.
    
    Parameters:
    - magnet_collection: A magpylib Collection object containing the magnets
    """
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111, projection='3d')
    
    # Add displaySystem on ax1
    magpy.show(magnet_collection, canvas=ax1, suppress=True)
    ax1.view_init(elev=20, azim=30)  # Set perspective to more side view
    
    plt.show()



# Vectorized function to evaluate the magnetic flux density at given points (r, z)
def calculate_flux(z, magnet, z_segments, disc_radius, r_discretization):
    """
    Calculate the magnetic flux through a set of discs at positions of z_segments.

    Args:
        z (float): The offset position of the magnet from the origin.
        magnet: The magnet object used to calculate the magnetic flux 
        z_segments (array-like): The array of z-coordinates for each disc that the flux is calculated through.
        disc_radius (float): The radius of the discs.
        r_discretization (int): An integer representing the granularity of discretization in the radial direction.
    
    Returns:
        array-like: An array containing the magnetic flux through each z and disc. Shape is (len(z), len(z_segments)).
    """

    # For efficiency we buld a tensor that contains the z-coordinates of the discs for each z position.
    #z_segments_aug = np.expand_dims(z_segments, axis=0) + z  # Add the z offset to each z-segment
    z_positions = ((z_segments[..., np.newaxis] - z).T).flatten()

    r_segments = np.linspace(0, disc_radius, r_discretization)  # Radial segments
    # Find the points between each radial segment.
    middle_r = (r_segments[1:] + r_segments[:-1]) / 2
    R, Z = np.meshgrid(middle_r, z_positions)  # Create a grid of r and z points
    X = R.flatten()
    Y = np.zeros_like(X)
    Z = Z.flatten()

    # Calculate the magnetic flux density at all points
    B = magnet.getB(np.column_stack((X, Y, Z)))

    Bz = B[:, 2]  # Extract the z-component of the magnetic flux density


    # The Bz entries start from the first z, z-segment and middle radius, visit each middle radius value in the z-segment, and then move to the next z-segment, visit each z-segment and the move to the next z.
    # We need to calculate the flux for each segment. 
    # Calculate the flux through each segment by multiplying the area of the annular sector by the magnetic flux density at the middle radius of the segment.    

    annular_areas = np.pi * (r_segments[1:] ** 2 - r_segments[:-1] ** 2)  # Area of each slice of the disc
    annular_areas = np.tile(annular_areas, len(z_positions)) # Repeat the annular areas for each z-segment. (These are the same for each z-segment.)
    flux_elements = annular_areas * Bz # Multiply the area by the magnetic flux density to get the flux through each annular sector.
    flux_through_segment = flux_elements.reshape((len(z), len(z_segments), len(middle_r))).sum(axis=2) # Sum the flux through each annular sector to get the total flux through each z-segment.

    
    return flux_through_segment


def plot(t, V, R, f = None):
    # Induced current (Ohm's Law)
    I = V / R

    # Power generated
    P = V * I

    print("Calculated Resistance: ", R)
    print("Constant Power Generated: ", P.mean())

    # Plotting the results
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, V)


    
    plt.title('Induced Voltage')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')

    plt.subplot(3, 1, 2)
    plt.title('Induced Current')
    plt.plot(t, I)
    plt.xlabel('Time (s)')
    plt.ylabel('Current (A)')

    plt.subplot(3, 1, 3)
    plt.plot(t, P)
    plt.title('Generated Power')
    plt.xlabel('Time (s)')
    plt.ylabel('Power (W)')

    if (f is not None):
        # Add a secondary plot that shows the aplitude of the motion.
        # A new axis will be needed because the magnitude of the movement is different to that of voltage.
        z = np.vectorize(f)(t) # Calculate the position of the magnet at each time step.
        ax2 = plt.gca().twinx()
        ax2.plot(t, z, 'r--')
        
    plt.tight_layout()
    plt.show()


def calculate_inner_radius(outer_radius, coil_length, number_of_turns, wire_diameter):
    # Calculate the number of turns per layer
    turns_per_layer = coil_length / wire_diameter
    
    # Initialize variables
    remaining_turns = number_of_turns
    current_radius = outer_radius
    wire_length = 0
    
    # Loop through each layer and calculate the inner radius
    while remaining_turns > 0:
        if current_radius <= 0:
            raise ValueError("Invalid configuration: The number of turns and wire diameter do not fit into the outer radius.")
        
        # Calculate the number of turns in the current layer
        turns_in_layer = min(remaining_turns, turns_per_layer)
        
        # Calculate the length of wire in the current layer
        layer_length = 2 * np.pi * current_radius * turns_in_layer
        wire_length += layer_length
        
        # Subtract the current layer's thickness from the radius
        current_radius -= wire_diameter
        
        # Update remaining turns
        remaining_turns -= turns_in_layer
    
    inner_radius = current_radius
    if inner_radius <= 0:
        raise ValueError("Invalid configuration: The number of turns and wire diameter do not fit into the outer radius.")
    
    return inner_radius, wire_length
    

def calculate_resistance_of_coil(wire_length, d_wire=0.0002, rho=1.68e-8):
    """
    Calculate the resistance of the coil based on the wire length.
    """
    wire_cross_section = np.pi * (d_wire / 2)**2
    R = rho * wire_length / wire_cross_section
    return R


def calculate_voltage(t, z_func, r, N, coil_length, magnet, r_discretization=3, z_discretization=4, coil_orientation=[]):
    """
    Calculate the voltage across the coil by summing up the voltages across the coil segments.
    
    Parameters:
    - t: Time array
    - z_func: Function that takes t and returns z
    - z_segments: Position array of the coil segments
    - r: Radius of the coil
    - m: Magnetic moment
    - mu0: Permeability of free space
    - N: Number of turns in the coil
    - coil_length: Length of the coil
    - segment_length: Length of the coil segment
    - magnet: The magnet object used to calculate the magnetic flux density
    - coil_orientation: An array indicating at what points the sign of the coil direction is flipped. Starts with +. Every entry in coil_orientation must be smaller than coil_length.
    
    Returns:
    - Voltage across the coil (array)
    """

    # Assert that coil_orientation is sorted and that all values are smaller than coil_length and positive
    assert all([0 <= x <= coil_length for x in coil_orientation]), "coil_orientation must be a list of positive values smaller than coil_length."
    assert all([x < y for x, y in zip(coil_orientation, coil_orientation[1:])]), "coil_orientation must be sorted."
    
    z = np.vectorize(z_func)(t)

    z_segments = np.linspace(-coil_length / 2, coil_length / 2, z_discretization)  # Segments along the z-axis
    segment_length = coil_length / z_discretization  # Length of each segment
    flux_through_segments = calculate_flux(z, magnet, z_segments, r, r_discretization)
    
    # Calculate the rate of change of magnetic flux with respect to time
    dB_dt = np.gradient(flux_through_segments, t, axis=0)
    
    # Calculate the induced electric field
    E = -dB_dt / (2 * np.pi * r)

    # Compile a numpy array based on the coil_orientation array
    field_sign = np.ones(E.shape[1])
    previous_end = 0
    for index, z in enumerate(coil_orientation):
        start_index = int(previous_end / coil_length * z_discretization)
        end_index = int(z / coil_length * z_discretization)
        previous_end = z

        if index % 2 == 0:
            field_sign[start_index:end_index] = 1
        else:
            field_sign[start_index:end_index] = -1

    # Apply the sign of the last segment
    if len(coil_orientation) % 2 == 1:
        field_sign[int(previous_end / coil_length * z_discretization):] = -1
    
    #print("Field sign: ", field_sign)
    # print("Field sign: ", field_sign)

    # evaluate_at = z_discretization // 3
    # # Plot the magnetic flux at a specific position to debug.
    # plt.figure(figsize=(12, 2))
    # plt.plot(t, flux_through_segments[:, evaluate_at])
    # plt.title(f"Magnetic Flux at z={z_segments[evaluate_at]}")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Magnetic Flux (T*m^2)")
    # plt.show()


    # # Plot the field signs
    # plt.figure(figsize=(12, 2))
    # plt.plot(z_segments, field_sign)
    # plt.title(f"Field Sign at z={z_segments[evaluate_at]}")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Field Sign")
    # plt.show()


    # # #print("Magnet was at z = ", z[evaluate_at])
    # # #Plot the electric field at a specific position to debug.
    # plt.figure(figsize=(12, 2))
    # plt.plot(t, E[:, evaluate_at])
    # plt.title(f"Electric Field at z={z_segments[evaluate_at]}")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Electric Field (V/m)")
    # plt.show()


    # Calculate the parallel component of the wire
    parallel_component_of_the_wire = 2 * np.pi * r * N / coil_length * segment_length
    
    # Calculate the total voltage across the coil
    V = np.sum(E * field_sign * parallel_component_of_the_wire, axis=1)
    
    return V