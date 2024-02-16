import serial
import time

# Set the serial port and baud rate
serial_port = 'COM3'  # Replace with the appropriate port name
baud_rate = 57600

# Open the serial connection
ser = serial.Serial(serial_port, baud_rate, timeout=1)

# Wait for the Arduino to initialize
time.sleep(2)

# Send numbers from 0 to 4095
for number in range(4095):
    # Send the number as a string
    ser.write(str(number).encode())
    print(f"Sent: {number}")
    time.sleep(0.01)
    
    response = ser.readline().decode().strip()
    print(f"Response: {response}")
    

# Close the serial connection
ser.close()