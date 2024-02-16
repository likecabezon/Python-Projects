import struct

def read_mca_tables(file_path):
    try:
        with open(file_path, 'rb') as mca_file:
            header = mca_file.read(8192)  # Read the first 8KB (header) of the .mca file
            
            # Initialize empty lists for the two tables
            offset_table = []
            timestamp_table = []
            
            # Read the offset table (first 4KB of the header)
            for i in range(1024):
                offset = struct.unpack_from('>I', header, i * 4)  # Read 4 bytes as a big-endian unsigned int
                offset_table.append(offset[0])
            
            # Read the timestamp table (next 4KB of the header)
            for i in range(1024):
                timestamp = struct.unpack_from('>I', header, 4096 + i * 4)  # Read 4 bytes as a big-endian unsigned int
                timestamp_table.append(timestamp[0])
            
            return offset_table, timestamp_table
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage:
file_path = 'C:\\Users\\Luis\\AppData\\Roaming\\.minecraft\\saves\\BRUNO SORIA\\region\\r.0.0.mca'
offsets, timestamps = read_mca_tables(file_path)

print(offsets)
print(timestamps)
# Now you can use 'offsets' and 'timestamps' as lists containing the tables.