import pygame as pg
import random, math
import config

class Paddle:
    def __init__(self, x):
        self.start_x = x
        self.start_y = (config.SCREEN_HEIGHT - config.PADDLE_HEIGHT) / 2

        self.width = config.PADDLE_WIDTH
        self.height = config.PADDLE_HEIGHT
        self.rect = pg.Rect(x, self.start_y, self.width, self.height)

        self.speed = config.PADDLE_SPEED
        self.color = (255, 255, 255)

    def move(self, action):
        if action == 1:
            self.rect.y -= self.speed
        else:
            self.rect.y += self.speed

        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > config.SCREEN_HEIGHT:
            self.rect.bottom = config.SCREEN_HEIGHT

    def reset(self):
        self.rect.x = self.start_x
        self.rect.y = self.start_y

    def draw(self, surface):
        pg.draw.rect(surface, self.color, self.rect)

class Ball:
    def __init__(self):
        self.radius = config.BALL_RADIUS
        self.color = (255, 255, 255)
        self.rect = pg.Rect(0, 0, self.radius * 2, self.radius * 2)

        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0

        self.reset()
    
    def reset(self):
        self.x = config.SCREEN_WIDTH / 2 - self.radius
        self.y = config.SCREEN_HEIGHT / 2 - self.radius
        self.rect.x = int(self.x)
        self.rect.y = int(self.y)

        angle = random.choice([
            random.uniform(-45, 45),
            random.uniform(135, 255)
        ])

        speed = config.BALL_INIT_SPEED
        self.vx = speed * math.cos(math.radians(angle))
        self.yx = speed * math.sin(math.radians(angle))

    def update(self, paddle_left, paddle_right):
        self.x += self.vx
        self.y += self.vy
        
        self.rect.x = int(self.x)
        self.rect.y = int(self.y)

        event = None

        if self.rect.top <= 0:
            self.rect.top = 0
            self.vy *= -1
        elif self.rect.bottom >= config.SCREEN_HEIGHT:
            self.rect.bottom = config.SCREEN_HEIGHT
            self.vy *= -1


        if self.rect.colliderect(paddle_left.rect):
            if self.vx < 0:
                self._handle_paddle_hit(paddle_left)
                event = "hit_left"
        elif self.rect.colliderect(paddle_right.rect):
            if self.vx > 0:
                self._handle_paddle_hit(paddle_right)
                event = "hit_right"
            
        return event
    
    def _handle_paddle_hit(self, paddle):
        self.vx *= -1.05

        if abs(self.vx) > config.BALL_MAX_SPEED:
             self.vx = config.BALL_MAX_SPEED * (-1 if self.vx < 0 else 1)

        offset = (self.rect.centery - paddle.rect.centery) / (paddle.height / 2)
        
        self.vy += offset * 2


    def draw(self, surface):
        pg.draw.rect(surface, self.color, self.rect)