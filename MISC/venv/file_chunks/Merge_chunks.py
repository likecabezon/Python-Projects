


def merge_files(file_names, output_file_name):
    with open(output_file_name, 'wb') as output_file:
        for file_name in file_names:
            with open(file_name, 'rb') as input_file:
                while chunk := input_file.read(1048576):
                    output_file.write(chunk)

# Example usage:
file_names = ['file1.bin', 'file2.bin', 'file3.bin']
output_file_name = 'merged_output.bin'
merge_files(file_names, output_file_name)
