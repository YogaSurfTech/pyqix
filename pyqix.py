import os

import pygame
from pygame.locals import *

WIDTH = 1024
HEIGHT = 1024
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 1024
X_SCALE = (WIDTH / 256.0)
Y_SCALE = (HEIGHT / 256.0)

TPS = 60  # number of game updates per second
SKIP_TICKS = 1000 / TPS  # ms to start skipping frames
MAX_FRAMESKIP = 5  # no we calc max updates (if we are behind) before displaying
MAX_TIMESKIP = 2000  # max Time we try to catch up until we just reset counter

screen = None
window_surface = None
frame_counter = 0  # counts the frames (updating qix not every frame)
logo = None


def hal_blt(img, coords):
    screen.blit(img, (coords[0] * X_SCALE, coords[1] * Y_SCALE))


def hal_load_image(fullname, color_key=None):
    try:
        image = pygame.image.load(fullname)
    except pygame.error as message:
        print('Cannot load image:', fullname)
        raise SystemExit(message)
    image = image.convert()
    if color_key is not None:
        if color_key == -1:
            color_key = image.get_at((0, 0))
        image.set_colorkey(color_key, RLEACCEL)
    return image, image.get_rect()


def get_real_time():
    """ Real time of the system in ms after pygame.init"""
    return pygame.time.get_ticks()


def paint_game():
    hal_blt(logo, (24, 16))


def init():
    global window_surface, screen, logo
    pygame.init()
    pygame.mixer.init()
    window_surface = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])
    screen = pygame.Surface((WIDTH, HEIGHT))
    logo, _ = hal_load_image(os.path.join('data', 'qix_logo.png'))
    logo = pygame.transform.scale(logo, (int(56.0 * X_SCALE), int(20 * Y_SCALE)))


def gameloop():  # https://dewitters.com/dewitters-gameloop/
    global frame_counter
    next_game_tick = get_real_time() - 1
    while 1:
        loops = 0
        while get_real_time() > next_game_tick and loops < MAX_FRAMESKIP:
            for event in pygame.event.get():
                if event.type == QUIT:
                    return
            frame_counter += 1
            next_game_tick += SKIP_TICKS
            if get_real_time() > next_game_tick + MAX_TIMESKIP:
                next_game_tick = get_real_time() + SKIP_TICKS
            loops += 1
        screen.fill(0)
        paint_game()
        window_tmp = pygame.transform.scale(screen, (WINDOW_WIDTH, WINDOW_HEIGHT))
        window_surface.blit(window_tmp, (0, 0))
        pygame.display.flip()


if __name__ == '__main__':
    print("pyQix")
    print("-----")
    print("(c) 2021 YogaSurfTech) https://github.com/YogaSurfTech/pyqix")
    print("A faithful remake of the Taito classic arcade game QIX in Python. ")
    print("On the occasion of the 40th anniversary of the release at 18th. October 2021")
    init()
    gameloop()
