import libe1701py
import time
import serial.tools.list_ports



ports = serial.tools.list_ports.comports()
for port in ports:
    print(f"{port.device} - {port.description}")



Cardnum = libe1701py.set_connection("/dev/ttyACM0")

print(f"Cardnum: {Cardnum}")

ret=libe1701py.load_correction(Cardnum,"",0); # set correction file, for no/neutral correction use "" or NULL here

print(f"correction data: {ret}")

time.sleep(0.1)  


info = libe1701py.get_card_info(Cardnum)
print(f"Info: {info}")

state = libe1701py.get_card_state(Cardnum)

print(f"State: {state}")
time.sleep(0.5) 


libe1701py.jump_abs(Cardnum, 0, 0, 0)
time.sleep(1)
#libe1701py.mark_abs(Cardnum, 134217728, 134217728, 0)  # moitié du max 28 bits
#time.sleep(1)



libe1701py.close(Cardnum)
