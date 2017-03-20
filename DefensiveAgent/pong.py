#!/usr/bin/env python
import numpy
import pygame
import os
from pygame.locals import *
from sys import exit
import random
from operator import add, sub
import pygame.surfarray as surfarray
import matplotlib.pyplot as plt

FPS = 30
position = 5, 325
pygame.init()
FPSCLOCK = pygame.time.Clock()
screen = pygame.display.set_mode((640,480),0,32)
back = pygame.Surface((640,480))
background = back.convert()
background.fill((0,0,0))
bar = pygame.Surface((10,50))
bar1 = bar.convert()
bar1.fill((255,255,255))
bar2 = bar.convert()
bar2.fill((255,255,255))
circ_sur = pygame.Surface((15,15))
circ = pygame.draw.rect(circ_sur, (255,255,255), (7,7,15,15))
circle = circ_sur.convert()
circle.set_colorkey((0,0,0))
font = pygame.font.SysFont("ArcadeClassic",40)

#Player switch
pswitch = False
#Perfect AI switch
qswitch = True
#Basic AI switch
bswitch = False

ai_switch = True
ai_speed = 20.

#Reward values
HIT_REWARD = 1
LOSE_REWARD = -3
SCORE_REWARD = 0

class GameState:
    def __init__(self):
        self.bar1_x, self.bar2_x = 10. , 620.
        self.bar1_y, self.bar2_y = 215. , 215.
        self.circle_x, self.circle_y = 307.5, 232.5
        self.bar1_move, self.bar2_move = 0. , 0.
        self.bar1_score, self.bar2_score = 0,0
        self.speed_x, self.speed_y = 7., 7.

    def frame_step(self,input_vect):
        pygame.event.pump()
        reward = 0

        if sum(input_vect) != 1:
            raise ValueError('Multiple input actions!')

        if input_vect[1] == 1:#Key up
            self.bar1_move = -ai_speed
        elif input_vect[2] == 1:#Key down
            self.bar1_move = ai_speed
        else: # don't move
            self.bar1_move = 0
                
        self.score1 = font.render(str(self.bar1_score), True,(255,255,255))
        self.score2 = font.render(str(self.bar2_score), True,(255,255,255))

        screen.blit(background,(0,0))
        frame = pygame.draw.rect(screen,(255,255,255),Rect((5,5),(630,470)),2)
        middle_line = pygame.draw.aaline(screen,(255,255,255),(330,5),(330,475))
        screen.blit(bar1,(self.bar1_x,self.bar1_y))
        screen.blit(bar2,(self.bar2_x,self.bar2_y))
        screen.blit(circle,(self.circle_x,self.circle_y))
        #screen.blit(self.score1,(250.,210.))
        #screen.blit(self.score2,(380.,210.))

        self.bar1_y += self.bar1_move
        self.bar2_y += self.bar2_move
        #global ai_switch
        #global ai_switch2
        global qswitch
        global pswitch
        global bswitch
        #Player Mode
        #Perfect AI Mode
        if qswitch == True:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_p:
                        if qswitch == True:
                            pswitch = True
                            bswitch = False
                            qswitch = False
                    elif event.key == K_b:
                        if qswitch == True:
                            pswitch = False
                            bswitch = True
                            qswitch = False
            if not self.bar2_y == self.circle_y:
                if self.bar2_y < self.circle_y:
                    self.bar2_y = self.circle_y
                if self.bar2_y > self.circle_y:
                    self.bar2_y = self.circle_y
        #Player Mode
        if pswitch == True:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_w:
                        self.bar2_move = -ai_speed
                    elif event.key == K_s:
                        self.bar2_move = ai_speed
                    elif event.key == K_q:
                        qswitch = True
                        bswitch = False
                        pswitch = False
                    elif event.key == K_b:
                        qswitch = False
                        bswitch = True
                        pswitch = False
                elif event.type == KEYUP:
                    if event.key == K_w:
                        self.bar2_move = 0.
                    elif event.key == K_s:
                        self.bar2_move = 0.
        #Basic AI Mode
        if bswitch == True:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_q:
                        if bswitch == True:
                            pswitch = False
                            qswitch = True
                            bswitch = False
                    if event.key == K_p:
                        if bswitch == True:
                            pswitch = True
                            bswitch = False
                            qswitch = False
            #If circle is in the basic AI's half of the court
            ops = (add, sub)
            ops = random.choice(ops)
            if self.circle_x <= 305.:
                self.bar2_y = ops(self.bar2_y, 8.0)
            if self.circle_x >= 205.:
                if not self.bar2_y == self.circle_y + 4.5:
                    if self.bar2_y < self.circle_y + 4.5:
                        self.bar2_y += ai_speed
                    if  self.bar2_y > self.circle_y - 36.5:
                        self.bar2_y -= ai_speed
                else:
                    self.bar2_y == self.circle_y + 4.5

        # bounds of movement
        if self.bar1_y >= 420.: self.bar1_y = 420.
        elif self.bar1_y <= 10. : self.bar1_y = 10.
        if self.bar2_y >= 420.: self.bar2_y = 420.
        elif self.bar2_y <= 10.: self.bar2_y = 10.

        #since i don't know anything about collision, ball hitting bars goes like this.
        if self.circle_x <= self.bar1_x + 10.:
            if self.circle_y >= self.bar1_y - 7.5 and self.circle_y <= self.bar1_y + 42.5:
                self.circle_x = 20.
                self.speed_x = -self.speed_x
                reward = HIT_REWARD

        if self.circle_x >= self.bar2_x - 15.:
            if self.circle_y >= self.bar2_y - 7.5 and self.circle_y <= self.bar2_y + 42.5:
                self.circle_x = 605.
                self.speed_x = -self.speed_x

        # scoring
        if self.circle_x < 5.:
            self.bar2_score += 1
            reward = LOSE_REWARD
            self.circle_x, self.circle_y = 320., 232.5
            self.bar1_y,self.bar_2_y = 215., 215.
        elif self.circle_x > 620.:
            self.bar1_score += 1
            reward = SCORE_REWARD
            self.circle_x, self.circle_y = 307.5, 232.5
            self.bar1_y, self.bar2_y = 215., 215.

        # collisions on sides
        if self.circle_y <= 10.:
            self.speed_y = -self.speed_y
            self.circle_y = 10.
        elif self.circle_y >= 457.5:
            self.speed_y = -self.speed_y
            self.circle_y = 457.5

        self.circle_x += self.speed_x
        self.circle_y += self.speed_y

        #Capture a 3d array of the game screen
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        pygame.display.update()
        FPSCLOCK.tick(FPS)
        #terminal is used as a flag that determines whether a
        #set is finished or not.
        #In pong the game is finished when a player has scored 20 points
        terminal = False
        if max(self.bar1_score, self.bar2_score) >= 20:
            self.bar1_score = 0
            self.bar2_score = 0
            terminal = True
        #Return the game screen, the current reward obtained and whether the set
        #has finished or not.
        return image_data, reward, terminal
