import pygame
from pygame.locals import *

WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 1024

window_surface = None


def init():
    global window_surface
    pygame.init()
    window_surface = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])


def gameloop():		# http://www.koonsolo.com/news/dewitters-gameloop/
    while 1:
        for event in pygame.event.get():
            if event.type == QUIT:
                return


if __name__ == '__main__':
    print("pyQix")
    print("-----")
    print("(c) 2021 YogaSurfTech https://github.com/YogaSurfTech/pyqix")
    print("A faithful remake of the Taito classic arcade game QIX in Python. ")
    print("On the occasion of the 40th anniversary of the release at 18th. October 2021")
    init()
    gameloop()

