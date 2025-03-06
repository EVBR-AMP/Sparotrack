import serial
import time

# Define the serial port and baud rate
SERIAL_PORT = '/dev/ttyAMA0'  # Explicitly use ttyAMA0 as the UART device
BAUD_RATE = 115200            # Common baud rate for testing

def test_serial_connection():
    try:
        # Initialize the serial connection
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            timeout=1  # 1-second timeout for reading
        )
        print(f"Serial port {SERIAL_PORT} opened successfully at {BAUD_RATE} baud.")

        # Test message to send
        test_message = "Hello, Raspberry Pi 5 UART!\n"
        print(f"Sending: {test_message.strip()}")

        # Send the test message
        ser.write(test_message.encode('utf-8'))
        time.sleep(0.1)  # Brief delay to ensure transmission

        # Check for a response (e.g., from loopback or external device)
        if ser.in_waiting > 0:
            response = ser.readline().decode('utf-8').strip()
            print(f"Received: {response}")
        else:
            print("No response received. Check loopback or device connection.")

    except serial.SerialException as e:
        print(f"Error with serial port: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Ensure the serial port is closed
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed.")

if __name__ == "__main__":
    test_serial_connection()