import struct
import numpy as np


def create_bmp_file_header(image_size,offset):
    # Constants for BMP file header
    FILE_SIGNATURE = b'BM'
    RESERVED1 = 0
    RESERVED2 = 0
    OFFSET = offset + 14
    # Calculate the file size (including header and pixel data)
    file_size = 14 + image_size

    # Create the BMP file header
    bmp_file_header = struct.pack('<2sIHHI', FILE_SIGNATURE, file_size, RESERVED1, RESERVED2, OFFSET)

    return bmp_file_header

def create_dib_header(image_width, image_height, horizontal_resolution, vertical_resolution,array_size):
    # Constants for BMP DIB header
    
    IMAGE_WIDTH = image_width
    IMAGE_HEIGHT = image_height
    PLANES = 1  # Number of color planes (must be 1)
    BITS_PER_PIXEL = 1
    COMPRESSION = 0  # Compression method (0 for uncompressed)
    IMAGE_SIZE = array_size  # Size of the raw bitmap data; a dummy 0 can be given for BI_RGB bitmaps
    HORIZONTAL_RESOLUTION = horizontal_resolution  # Horizontal resolution of the image (pixel per meter, signed integer)
    VERTICAL_RESOLUTION = vertical_resolution  # Vertical resolution of the image (pixel per meter, signed integer)
    COLORS_IN_COLOR_TABLE = 2  # Number of colors in the color palette, or 0 to default to 2^n
    IMPORTANT_COLOR_COUNT = 0  # Number of important colors used, or 0 when every color is important; generally ignored
    

    # Create the BMP DIB header
    dib_header = struct.pack(
        '<iiHHIIIIII',
        IMAGE_WIDTH,  # Bitmap width in pixels (signed integer)
        IMAGE_HEIGHT,  # Bitmap height in pixels (signed integer)
        PLANES,  # Number of color planes (must be 1)
        BITS_PER_PIXEL,  # Bits per pixel (color depth)
        COMPRESSION,  # Compression method
        IMAGE_SIZE,  # Image size (dummy value for BI_RGB bitmaps)
        HORIZONTAL_RESOLUTION,  # Horizontal resolution
        VERTICAL_RESOLUTION,  # Vertical resolution
        COLORS_IN_COLOR_TABLE,  # Number of colors in the color palette
        IMPORTANT_COLOR_COUNT  # Number of important colors
    )
    DIB_HEADER_SIZE = len(dib_header) + 4  # Size of the DIB header
    dib_header = struct.pack('<I', DIB_HEADER_SIZE) + dib_header

    return dib_header, DIB_HEADER_SIZE 


def create_bw_color_table():
    # Black and white color table
    color_table = struct.pack('<8B', 0, 0, 0, 0, 255, 255, 255, 0)
    return color_table, len(color_table)

def create_bmp_pixel_array(bool_array):
    # Get the dimensions of the boolean array
    height, width = bool_array.shape

    # Calculate the row size in bytes (including padding)
    row_size_bytes = ((width + 31) // 32) * 4

    # Initialize an empty byte array to store the pixel data
    pixel_data = bytearray()

    # Iterate through the rows of the boolean array
    for y in range(height):
        row_data = bytearray()
        current_byte = 0
        bits_written = 0
        for x in range(width):
            # Pack multiple 1-bit pixels into a single byte
            pixel_value = bool_array[y, x]
            current_byte = (current_byte << 1) | pixel_value
            bits_written += 1
            if bits_written == 8:
                row_data.append(current_byte)
                current_byte = 0
                bits_written = 0

        # If there are remaining bits, add them
        if bits_written > 0:
            current_byte = current_byte << (8 - bits_written)
            row_data.append(current_byte)

        # Calculate the number of padding bytes needed to align to 4-byte boundaries
        padding_bytes = row_size_bytes - len(row_data)

        # Add the row data to the pixel data
        pixel_data.extend(row_data)

        # Add padding bytes if necessary
        if padding_bytes > 0:
            pixel_data.extend(bytes(padding_bytes))

    return bytes(pixel_data), row_size_bytes * height

def create_bmp_file(pixel_array, output_filename, horizontal_ppm, vertical_ppm):
    height, width = pixel_array.shape
    pixel_array_data,pixel_array_size = create_bmp_pixel_array(pixel_array)
    color_table_data,color_table_size = create_bw_color_table()
    dib_header_data, dib_header_size = create_dib_header(width, height, horizontal_ppm, vertical_ppm, pixel_array_size)
    header_data = create_bmp_file_header(pixel_array_size + color_table_size + dib_header_size, color_table_size + dib_header_size)
    
    # Concatenate the BMP header, DIB header, color table, and pixel array data
    bmp_file_data = header_data + dib_header_data + color_table_data + pixel_array_data

    # Write the combined data to a BMP file
    with open(output_filename, 'wb') as bmp_file:
        bmp_file.write(bmp_file_data)


def create_rectangle(arr):
    if arr.shape != (500, 500):
        raise ValueError("Input array must be of shape (500, 500)")

    # Create a copy of the input array
    result = arr.copy()

    # Define the coordinates of the top-left corner and bottom-right corner of the rectangle
    top_left = (100, 190)  # Adjust these values as needed
    bottom_right = (top_left[0] + 79, top_left[1] + 119)

    # Set the values in the rectangle to True
    result[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1] = True

    return result

def read_bmp_file_header(file_path):
    # Open the BMP file
    with open(file_path, 'rb') as file:
        # Read the BMP File Header (14 bytes)
        file_header_format = '<2sIHHI'
        file_header_data = file.read(14)
        if len(file_header_data) != 14:
            raise ValueError("Invalid BMP file header size")
        signature, file_size, _, _, pixel_data_offset = struct.unpack(file_header_format, file_header_data)
        if signature != b'BM':
            raise ValueError("Invalid BMP file signature")
        return file_size, pixel_data_offset

def read_dib_header(file_path):
    # Open the BMP file
    with open(file_path, 'rb') as file:
        # Read and discard the BMP File Header (14 bytes)
        file.read(14)
        
        # Read the DIB Header (40 bytes)
        dib_header_format = '<IiiHHIIIIII'
        dib_header_data = file.read(40)
        
        if len(dib_header_data) != 40:
            raise ValueError("Invalid DIB header size")
        
        header_size, image_width, image_height, _, bits_per_pixel, _, _, _, _, _, _ = struct.unpack(dib_header_format, dib_header_data)
        
        if header_size != 40:
            raise ValueError("Unsupported DIB header size")
        
        return image_width, image_height, bits_per_pixel
    

def read_bmp_bw_matrix(file_name, image_width, image_height, bits_per_pixel, matrix_offset):
    if bits_per_pixel != 1:
        raise ValueError("Unsupported bits per pixel value. Expected 1 bpp.")

    with open(file_name, 'rb') as file:
        # Calculate the row size in bytes (including padding)
        row_size_bytes = ((image_width + 31) // 32) * 4

        file.seek(matrix_offset)  # Seek to the specified matrix offset

        # Read the pixel data starting from the specified offset
        pixel_data = file.read()

        if  len(pixel_data) != row_size_bytes * image_height:
            raise ValueError("Invalid pixel data size or offset")

        # Create a NumPy array to store the B/W matrix
        bw_matrix = np.empty((image_height, image_width), dtype=bool)

        # Iterate through the pixel data to populate the matrix
        row_start = 0
        for y in range(image_height):
            for x in range(image_width):
                byte_offset = x // 8
                bit_offset = 7 - (x % 8)
                pixel_byte = pixel_data[row_start + byte_offset]
                bw_matrix[y, x] = (pixel_byte & (1 << bit_offset)) != 0
            row_start += row_size_bytes

    return bw_matrix




array = np.zeros((500,500),bool)
array = create_rectangle(array)
create_bmp_file(array,'output.bmp',50000,50000)

image_size, data_offset= read_bmp_file_header('output.bmp')
print(data_offset)
image_width,image_height,bpp = read_dib_header('output.bmp')
print(image_width)
print(image_height)
print(bpp)
array1 = read_bmp_bw_matrix('output.bmp',image_width,image_height,bpp,data_offset)
print(array1)
create_bmp_file(array1,'output1.bmp',50000,50000)