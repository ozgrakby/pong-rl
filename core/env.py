import pygame as pg
import numpy as np
import config
from core.entities import Ball, Paddle

class PongEnv:
    def __init__(self, render_mode = False):
        self.render_mode = render_mode
        
        self.width = config.SCREEN_WIDTH
        self.height = config.SCREEN_HEIGHT

        self.score_l = 0
        self.score_r = 0

        pg.init()
        self.pg = pg

        if self.render_mode:
            self.screen = pg.display.set_mode((self.width, self.height))
            self.clock = pg.time.Clock()
            pg.display.set_caption("Pong RL")

            self.font = pg.font.Font(None, 74)
        else:
            self.screen = pg.Surface((self.width, self.height))
        
        self.ball = None
        self.paddle_left = None
        self.paddle_right = None
        self.reset()
    
    def reset(self):
        self.ball = Ball()
        self.paddle_left = Paddle(50)
        self.paddle_right = Paddle(config.SCREEN_WIDTH - 50 - config.PADDLE_WIDTH)

        
    
        return self.get_state()
    
    def step(self, action_l, action_r):
        self.paddle_left.move(action_l)
        self.paddle_right.move(action_r)

        hit_result = self.ball.update(self.paddle_left, self.paddle_right)

        reward_l = 0
        reward_r = 0
        done = False

        if hit_result == "hit_left":
            reward_l += 0.1
        if hit_result == "hit_right":
            reward_r += 0.1

        if self.ball.x < 0:
            reward_l = -1
            reward_r = 1
            self.score_r += 1
            done = True
        elif self.ball.x > self.width:
            reward_l = 1
            reward_r = -1
            self.score_l += 1
            done = True
        
        if self.render_mode:
            self._draw_frame()
        
        next_state = self.get_state()
        return next_state, (reward_l, reward_r), done
    
    def get_state(self):
        state = [
            self.ball.x / self.width,
            self.ball.y / self.height,
            self.ball.vx / config.BALL_MAX_SPEED,
            self.ball.vy / config.BALL_MAX_SPEED,
            self.paddle_left.rect.y / self.height,
            self.paddle_right.rect.y / self.height
        ]
        return np.array(state, dtype=np.float32)
    
    def _draw_frame(self):
        self.screen.fill((0, 0, 0))

        dash_width = 5
        dash_height = 15
        gap = 25
        center_x = self.width // 2

        for y in range(0, self.height, gap):
            if y % (gap*2) == 0:
                 pg.draw.rect(self.screen, (255, 255, 255), (center_x - dash_width//2, y, dash_width, dash_height))

        text_l = self.font.render(str(self.score_l), True, (255, 255, 255))
        text_r = self.font.render(str(self.score_r), True, (255, 255, 255))
        
        rect_l = text_l.get_rect(center=(self.width // 4, 50))
        rect_r = text_r.get_rect(center=(self.width * 3 // 4, 50))
        
        self.screen.blit(text_l, rect_l)
        self.screen.blit(text_r, rect_r)

        self.paddle_left.draw(self.screen)
        self.paddle_right.draw(self.screen)
        self.ball.draw(self.screen)
        
        pg.display.flip()
        self.clock.tick(config.FPS) 

    def close(self):
        pg.quit()