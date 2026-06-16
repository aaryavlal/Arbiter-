
from time import sleep

from gpiozero import Servo

servo = Servo(18)  

values = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

for value in values:
    print(f"servo.value = {value}")
    servo.value = value
    sleep(2)

servo.detach()  
