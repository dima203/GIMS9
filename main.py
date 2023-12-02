from typing import Self
from random import randint
from math import floor, ceil
from pathlib import Path
import time

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

from OpenGL.GL import shaders

import freetype
import glm

import numpy as np


WINDOW_SIZE = (1920, 1080)
FPS = 60
SCORE = 0
TIME = 0
FONTS_DIR = Path(r'.\fonts')
FONTS = []
CHARACTERS_TEXTURES = []
if FONTS_DIR.is_dir():
    for file in FONTS_DIR.iterdir():
        FONTS.append(file)


class NumberSurface:
    number: int
    x: int
    y: int
    width: int
    height: int
    scale: float
    font: int
    color: tuple[int, ...]

    def __init__(self, number: int, x: int, y: int, scale: float = 1, font: int = 0,
                 color: tuple[int, ...] = (255, 255, 255)) -> None:
        self.number = number
        self.x = x
        self.y = y
        self.scale = scale
        self.font = font
        self.color = color
        self.width, self.height = self.__calculate_size()

    def draw(self) -> None:
        render_text(str(self.number), self.x, self.y, self.scale, self.font, self.color)

    def check_overlap(self, other: Self) -> bool:
        return not ((self.y - self.height) > other.y or self.y < (other.y - other.height) or
                    (self.x + self.width) < other.x or self.x > (other.x + other.width))

    def check_click(self, mouse_x: int, mouse_y: int) -> bool:
        return self.x < mouse_x < self.x + self.width and self.y > mouse_y > self.y - self.height

    def __calculate_size(self) -> tuple[int, int]:
        x = self.x
        for ch in str(self.number)[:-1]:
            x += (CHARACTERS_TEXTURES[self.font][ch].advance >> 6) * self.scale
        w, h = CHARACTERS_TEXTURES[self.font][str(self.number)[-1]].textureSize
        w, h = w * self.scale, h * self.scale
        end_character_data = _get_rendering_buffer(x, self.y, w, h)
        end_pos = ceil(end_character_data[20]), floor(end_character_data[21])
        return int(end_pos[0] - self.x), int(self.y - end_pos[1])

    def __repr__(self) -> str:
        return f'Number({self.number}, {self.x} - {self.x + self.width}, {self.y - self.height} - {self.y}, {self.scale})'


class CharacterSlot:
    def __init__(self, texture, glyph):
        self.texture = texture
        self.textureSize = (glyph.bitmap.width, glyph.bitmap.rows)

        if isinstance(glyph, freetype.GlyphSlot):
            self.bearing = (glyph.bitmap_left, glyph.bitmap_top)
            self.advance = glyph.advance.x
        elif isinstance(glyph, freetype.BitmapGlyph):
            self.bearing = (glyph.left, glyph.top)
            self.advance = None
        else:
            raise RuntimeError('unknown glyph type')


def _get_rendering_buffer(xpos, ypos, w, h):
    return np.asarray([
        xpos, ypos - h, 0, 0,
        xpos, ypos, 0, 1,
              xpos + w, ypos, 1, 1,
        xpos, ypos - h, 0, 0,
              xpos + w, ypos, 1, 1,
              xpos + w, ypos - h, 1, 0
    ], np.float32)


VERTEX_SHADER = """
        #version 330 core
        layout (location = 0) in vec4 vertex; // <vec2 pos, vec2 tex>
        out vec2 TexCoords;

        uniform mat4 projection;

        void main()
        {
            gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
            TexCoords = vertex.zw;
        }
       """

FRAGMENT_SHADER = """
        #version 330 core
        in vec2 TexCoords;
        out vec4 color;

        uniform sampler2D text;
        uniform vec3 textColor;

        void main()
        {    
            vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);
            color = vec4(textColor, 1.0) * sampled;
        }
        """

shaderProgram = None
VBO = None
VAO = None


def add_font_texture(font: str, textures_list: list[dict[str: CharacterSlot]]) -> None:
    face = freetype.Face(font)
    face.set_char_size(48 * 64)
    textures_list.append({})

    # load first 128 characters of ASCII set
    for i in range(0, 128):
        face.load_char(chr(i))
        glyph = face.glyph

        # generate texture
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, glyph.bitmap.width, glyph.bitmap.rows, 0,
                     GL_RED, GL_UNSIGNED_BYTE, glyph.bitmap.buffer)

        # texture options
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # now store character for later use
        textures_list[-1][chr(i)] = CharacterSlot(texture, glyph)


def initialize(window_size: tuple[int, int]):
    global VERTEXT_SHADER
    global FRAGMENT_SHADER
    global FONTS
    global CHARACTERS_TEXTURES
    global shaderProgram
    global VBO
    global VAO

    # compiling shaders
    vertexshader = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fragmentshader = shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)

    # creating shaderProgram
    shaderProgram = shaders.compileProgram(vertexshader, fragmentshader)
    glUseProgram(shaderProgram)

    # get projection
    # problem

    shader_projection = glGetUniformLocation(shaderProgram, "projection")
    projection = glm.ortho(0, *window_size, 0)
    glUniformMatrix4fv(shader_projection, 1, GL_FALSE, glm.value_ptr(projection))

    # disable byte-alignment restriction
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    for font in FONTS:
        add_font_texture(str(font), CHARACTERS_TEXTURES)

    glBindTexture(GL_TEXTURE_2D, 0)

    # configure VAO/VBO for texture quads
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, 6 * 4 * 4, None, GL_DYNAMIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)


def render_text(text, x, y, scale: float = 1, font: int = 0, color=(255, 255, 255)):
    global shaderProgram
    global VBO
    global VAO

    face = freetype.Face(str(FONTS[font]))
    face.set_char_size(48 * 64)
    glUniform3f(glGetUniformLocation(shaderProgram, "textColor"),
                color[0] / 255, color[1] / 255, color[2] / 255)

    glActiveTexture(GL_TEXTURE0)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glBindVertexArray(VAO)
    for i, c in enumerate(text):
        ch = CHARACTERS_TEXTURES[font][c]
        w, h = ch.textureSize
        w = w * scale
        h = h * scale
        vertices = _get_rendering_buffer(x, y, w, h)

        # render glyph texture over quad
        glBindTexture(GL_TEXTURE_2D, ch.texture)
        # update content of VBO memory
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        # render quad
        glDrawArrays(GL_TRIANGLES, 0, 6)
        # now advance cursors for next glyph (note that advance is number of 1/64 pixels)
        x += (ch.advance >> 6) * scale

    glBindVertexArray(0)
    glBindTexture(GL_TEXTURE_2D, 0)


def game_end() -> None:
    global SCORE

    SCORE *= 5 * 60 / TIME
    s = TIME // 1000
    m, s = divmod(TIME // 1000, 60)
    h, m = divmod(m, 60)
    print(int(SCORE), '\t', f'{h:02}:{m:02}:{s:02}')
    pygame.quit()
    quit()


def main():
    global SCORE
    global TIME

    pygame.init()
    pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF | OPENGL)

    clock = pygame.time.Clock()

    initialize(WINDOW_SIZE)

    numbers: list[NumberSurface] = []
    for i in range(1, 101):
        is_overlapping = True
        while is_overlapping:
            is_overlapping = False
            scale = randint(5, 30) / 10
            font = randint(0, len(FONTS) - 1)
            color = (randint(100, 255), randint(100, 255), randint(100, 255))
            number = NumberSurface(i, randint(0, WINDOW_SIZE[0] - int(70 * scale)),
                                   randint(int(40 * scale), WINDOW_SIZE[1]), scale, font, color)
            for n in numbers:
                if n.check_overlap(number):
                    is_overlapping = True
                    break
        numbers.append(number)

    last_removed = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    for i in range(len(numbers)):
                        if numbers[i].check_click(*pygame.mouse.get_pos()):
                            if numbers[i].number == last_removed + 1:
                                del numbers[i]
                                last_removed += 1
                                SCORE += 1000
                                if last_removed == 100:
                                    game_end()
                            else:
                                SCORE -= 200
                            break
                    else:
                        SCORE -= 50

        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        for number in numbers:
            number.draw()
        pygame.display.flip()
        clock.tick(FPS)
        TIME += clock.get_time()


if __name__ == '__main__':
    main()
