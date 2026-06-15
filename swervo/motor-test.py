from gpiozero.pins.lgpio import LGPIOFactory
from gpiozero import Servo
from time import sleep

factory = LGPIOFactory()
s = Servo(12, pin_factory=factory)

while True:
    s.min(); sleep(1)
    s.mid(); sleep(1)
    s.max(); sleep(1)