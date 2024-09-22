import os
import requests

CHUNK_SIZE =100 * 1024 * 1024  # 100 MB chunk size

def download_chunk(url, index):
    try:
        # Send a HEAD request to get the content length
        response = requests.head(url)
        total_size = int(response.headers.get('Content-Length', 0))
        print(f"Total file size: {total_size} bytes")

        # Calculate number of chunks needed
        num_chunks = (total_size + CHUNK_SIZE - 1) // CHUNK_SIZE  # Ceiling division
        print(f"Total number of chunks needed: {num_chunks}")

        # Check if the requested index is within bounds
        if index <= 0 or index > num_chunks:
            print(f"Error: Invalid index. Index should be between 1 and {num_chunks}.")
            return
        
        # Calculate start and end byte range for the requested chunk
        start = (index - 1) * CHUNK_SIZE
        end = min(index * CHUNK_SIZE - 1, total_size - 1)

        # Send GET request with Range header to download the chunk
        headers = {'Range': f'bytes={start}-{end}'}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        # Save chunk to file
        chunk_filename = f"chunk{index}.bin"
        with open(chunk_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        
        print(f"Chunk {index} downloaded and saved as {chunk_filename}.")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    url = input("URL:\n")  # Replace with your URL
    chunk_index = int(input("Enter The chunk you want:\n"))  # Replace with the desired chunk index

    download_chunk(url, chunk_index)
