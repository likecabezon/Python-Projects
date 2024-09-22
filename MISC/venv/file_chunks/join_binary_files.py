import sys

def join_binary_files(input_files, output_file):
    with open(output_file, 'wb') as outfile:
        for fname in input_files:
            with open(fname, 'rb') as infile:
                outfile.write(infile.read())

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python join_binary_files.py <input_file1> <input_file2> <input_file3> <input_file4> <input_file5> <output_file>")
        sys.exit(1)
    
    input_files = sys.argv[1:6]
    output_file = sys.argv[6]

    join_binary_files(input_files, output_file)
    print(f"Successfully joined files into {output_file}")
