import pigpio
import time

pi = pigpio.pi()                          # connect to the pigpiod daemon
SERVO = 18                                # GPIO18 = physical pin 12

pi.set_servo_pulsewidth(SERVO, 1500)      # center (~90°)
time.sleep(1)
pi.set_servo_pulsewidth(SERVO, 1000)      # one extreme (~0°)
time.sleep(1)
pi.set_servo_pulsewidth(SERVO, 2000)      # other extreme (~180°)
time.sleep(1)
pi.set_servo_pulsewidth(SERVO, 0)         # stop sending pulses (servo goes limp)

pi.stop()                                 # disconnect