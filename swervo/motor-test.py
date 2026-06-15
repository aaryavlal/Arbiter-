# gpiozero (easiest for servos)
from gpiozero import Servo
servo = Servo(18)   # the 18 means GPIO18

# RPi.GPIO
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)

# pigpio (least jitter, recommended for servos)
import pigpio
pi = pigpio.pi()
pi.set_servo_pulsewidth(18, 1500)   # 1500µs = center