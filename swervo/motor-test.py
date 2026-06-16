from gpiozero import Servo
from time import sleep

servo = Servo(18)  # GPIO18 = physical pin 12

servo.mid()        # center
sleep(1)
servo.min()        # one extreme
sleep(1)
servo.max()        # other extreme
sleep(1)
servo.detach()     # stop sending pulses (servo goes limp)