import math



def parse_mma_file(file_path):
    title = ""
    center_freq = 0.0
    wires = []
    sources = []
    loads = []
    segmentation = []
    parameters = []
    comment = ""
    

    with open(file_path, 'r') as file:
        lines = file.readlines()

        title = lines[0]

        # Parse center frequency
        center_freq = float(lines[2].strip())

        # Parse wires
        num_wires = int(lines[4].strip())
        wire_lines = lines[5:5 + num_wires]
        wires = [list(map(float, line.split(','))) for line in wire_lines]
        

        # Parse sources
        num_sources = int(lines[6 + num_wires].split(',')[0])
        source_lines = lines[7 + num_wires:7 + num_wires + num_sources]
        sources = [list(map(lambda x: x.strip(), line.split(','))) for line in source_lines]

        # Parse loads
        num_loads = int(lines[8 + num_wires + num_sources].split(',')[0])

        # Parse segmentation
        segmentation = list(map(float, lines[10 + num_wires + num_sources + num_loads].split(',')))

        # Parse parameters
        parameters = list(map(float, lines[12 + num_wires + num_sources + num_loads].split(',')))

        # Parse comment
        if(len(lines) > 13 + num_wires + num_sources + num_loads):
            comment = lines[-1].strip()

    return title, center_freq, wires, sources, loads, segmentation, parameters, comment

def create_mma_file(title, center_freq, wires, sources, loads, segmentation, parameters, comment, file_path):
    with open(file_path, 'w') as file:
        # Write title
        file.write(title)
        
        # Write center frequency
        file.write(f"*\n{center_freq}\n")
        
        # Write wires
        file.write("***Wires***\n")
        file.write(f"{len(wires)}\n")
        for wire in wires:
            file.write(','.join(map(str, wire)) + '\n')
        
        # Write source
        file.write("*** Source ***\n")
        file.write(f"{len(sources)}, 1\n")
        for source in sources:
            file.write(','.join(map(str, source)) + '\n')
        
        # Write load
        file.write("*** Load ***\n")
        file.write(f"{len(loads)}, 1\n")
        
        # Write segmentation
        file.write("*** Segmentation ***\n")
        file.write(','.join(map(str, segmentation)) + '\n')
        
        # Write G/H/M/R/AzEl/X
        file.write("*** G/H/M/R/AzEl/X ***\n")
        file.write(','.join(map(str, parameters)) + '\n')
        
        # Write comment
        file.write("### Comment ###\n")
        file.write(f"{comment}\n")

def create_can_rods(radius, num_rods, lenght, num_rings):
    wires = []

    # Create wires in X direction
    for i in range(num_rods):
        theta = i * (2 * math.pi / num_rods)
        x1= radius * math.cos(theta)
        x2 = x1
        y1 = radius * math.sin(theta)
        y2 = y1
        z1 = 0.0
        z2 = lenght

        wire = [x1, y1, z1, x2, y2, z2, 0.001, -1]  # Adjust wire radius (0.001) as needed
        wires.append(wire)
    
    for i in range(num_rings):
        for j in range(num_rods):
            theta = j * (2 * math.pi / num_rods)
            theta1 = (j + 1) * (2 * math.pi / num_rods)
            x1 = radius * math.cos(theta)
            x2 = radius * math.cos(theta1)
            y1 = radius * math.sin(theta)
            y2 = radius * math.sin(theta1)
            z1 = (i + 1) * lenght/(num_rings + 1)
            z2 = z1
            wire = [x1, y1, z1, x2, y2, z2, 0.001, -1]  # Adjust wire radius (0.001) as needed
            wires.append(wire)

    return wires
# Example usage
#parsed_data = parse_mma_file("C:\\MMANA-GALBasic3\\ANT\\VHF beams\\mirror.maa")
#print(parsed_data)
#create_mma_file(parsed_data[0], parsed_data[1], parsed_data[2], parsed_data[3], parsed_data[4], parsed_data[5], parsed_data[6], parsed_data[7], "output.maa")

name = "Cantenna 2.4Ghz\n"
freq = 2450
wires = create_can_rods(0.05,26,0.2,22)
sources = [["w2c",	'0.0',	'1.0']]
Loads =[]
segmentation = [100, 20 , 2, 1]
extra_data = [0, 20, 0, 50, 120, 60, 0]
description =""
print(len(wires))
create_mma_file(name,freq,wires,sources,Loads,segmentation,extra_data,description,"output.maa")
