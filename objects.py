import random
import pygame as pg
import math
from settings import *
from os import path

vec = pg.Vector2

my_font = path.join("PixelatedRegular-aLKm.ttf")


def draw_text(surf, text, font_name, size, color, x, y, align="nw"):
    font = pg.font.Font(font_name, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    if align == "nw":
        text_rect.topleft = (x, y)
    if align == "ne":
        text_rect.topright = (x, y)
    if align == "sw":
        text_rect.bottomleft = (x, y)
    if align == "se":
        text_rect.bottomright = (x, y)
    if align == "n":
        text_rect.midtop = (x, y)
    if align == "s":
        text_rect.midbottom = (x, y)
    if align == "e":
        text_rect.midright = (x, y)
    if align == "w":
        text_rect.midleft = (x, y)
    if align == "center":
        text_rect.center = (x, y)
    surf.blit(text_surface, text_rect)
    return text_rect


class Input:
    def __init__(self, pos, rect_half_diag):
        self.pos = vec(pos)
        self.val = 0
        self.clicked = False
        self.radius = rect_half_diag
        self.rect = pg.Rect(self.pos.x - self.radius, self.pos.y - self.radius, self.radius * 2, self.radius * 2)
        self.text = ''
        self.all_pressed = {}
        self.error = 0
        self.errors_accumulated = {}
        self.connection_to_neuron_after = []
        self.activation_function = lambda x: math.tanh(x)
        self.derivative_activation_function = lambda x: 1 - (math.tanh(x)) ** 2
        self.bias = random.uniform(-1, 1)
    def update(self):
        self.error = sum(self.errors_accumulated.values())
        if self.rect.collidepoint(pg.mouse.get_pos()) and pg.mouse.get_pressed()[0]:
            self.clicked = True
            self.text = str(self.val)
        if self.clicked:
            # input
            allowed_inputs = {x: pg.key.key_code(x) for x in "1234567890.-"}
            allowed_inputs["backspace"] = pg.key.key_code("backspace")
            allowed_inputs["return"] = pg.key.key_code("return")
            touch = pg.key.get_pressed()
            for (n, value) in allowed_inputs.items():
                if touch[value]:
                    if n not in self.all_pressed.keys():
                        if n == "backspace":
                            self.text = self.text[:-1]
                            self.all_pressed[n] = value
                        elif n == "return":
                            if self.text:
                                self.val = float(self.text)
                            else:
                                self.val = 0
                            self.clicked = False
                        else:
                            self.text += n
                            self.all_pressed[n] = value

            all_pressed_copy = self.all_pressed.copy()
            for (n, value) in all_pressed_copy.items():
                if not touch[value]:
                    self.all_pressed.pop(n)

    def adjust_connection_weight(self):
        for connection in self.connection_to_neuron_after:
            delta = LEARNING_RATE * self.error * self.derivative_activation_function(connection.neuron1.val) * connection.neuron1.val
            connection.weight += delta
            connection.neuron1.bias += LEARNING_RATE * self.error * self.derivative_activation_function(self.val)

    def set_weight(self, w):
        pass

    def draw(self, surf):
        pg.draw.rect(surf, WHITE, self.rect)
        if not self.clicked:
            r = draw_text(surf, str(self.val), my_font, self.radius * 2, BLACK, self.pos.x, self.pos.y, align="e")
            self.rect.width = max(r.width, self.radius * 2)
            self.rect.midright = r.midright
        else:
            r = draw_text(surf, self.text, my_font, self.radius * 2, RED, self.pos.x, self.pos.y, align="e")
            self.rect.width = max(r.width, self.radius * 2)
            self.rect.midright = r.midright

    def offset(self, offset):
        self.pos += vec(offset)


class Neuron:
    def __init__(self, bias, pos, radius, layer):
        self.pos = vec(pos)
        self.radius = radius
        self.bias = bias
        self.connections = []
        self.neurons_before = []
        self.connection_to_neuron_after = []
        self.errors_accumulated = {}
        self.error = 0
        # ReLU (Rectified Linear Unit) Activation Function
        # self.activation_function = lambda x : max(0, x)
        # self.activation_function = lambda x : 1 / (1 + math.exp(-x))
        # self.derivative_activation_function = lambda x: self.activation_function(x)*(1 - self.activation_function(x))
        self.activation_function = lambda x: math.tanh(x)
        self.derivative_activation_function = lambda x: 1 - (math.tanh(x)) ** 2
        self.val = self.output()
        self.layer = layer

    def update(self):

        self.error = sum(self.errors_accumulated.values())

        self.val = self.output()
        # distribute the error between neurons before
        all_weights = [c.weight for c in self.connections]
        # not recommended because sum of all weights can get to 0
        # all_weights_sum = sum(all_weights)
        # all_weights_percents = [w/all_weights_sum for w in all_weights]
        for n in range(len(self.neurons_before)):
            self.neurons_before[n].errors_accumulated[self] = all_weights[n] * self.error

    def adjust_connection_weight(self):
        # for connection in self.connection_to_neuron_after:
        #     delta = LEARNING_RATE * self.error * self.derivative_activation_function(self.val) * connection.neuron1.val
        #     connection.weight += delta
        #     connection.neuron1.bias += LEARNING_RATE * self.error * self.derivative_activation_function(self.val)
        for con in self.connections:
            delta = LEARNING_RATE * self.error * self.derivative_activation_function(self.val) * con.neuron2.val
            con.weight += delta
            con.neuron2.bias += LEARNING_RATE * self.error * self.derivative_activation_function(self.val)

    def set_weight(self, weight):
        self.connection_weight = weight

    def add_neuron_before(self, n):
        self.neurons_before.append(n)

    def get_neuron_before(self):
        return self.neurons_before

    def output(self):
        weighted_sum = 0
        for n in range(len(self.neurons_before)):
            weighted_sum += self.neurons_before[n].val * self.connections[n].weight
        return self.activation_function(weighted_sum + self.bias)


    def draw(self, surf):
        pg.draw.circle(surf, WHITE, self.pos, self.radius)
        draw_text(surf, f'{self.val:.2f}', my_font, self.radius * 2, GREEN, self.pos.x, self.pos.y, align="center")

    def draw_error(self, surf):
        pg.draw.circle(surf, WHITE, self.pos, self.radius)
        draw_text(surf, f'{self.error:.2f}', my_font, self.radius * 2, GREEN, self.pos.x, self.pos.y, align="center")

    def offset(self, offset):
        self.pos += vec(offset)


class Button:
    def __init__(self, pos, text_on, text_off):
        self.position = vec(pos)
        self.rect = pg.Rect(self.position.x - 50, self.position.y - 25, 100, 50)
        self.clicked = False
        self.click = False
        self.text_on = text_on
        self.text_off = text_off
    def update(self):
        if self.rect.collidepoint(pg.mouse.get_pos()) and pg.mouse.get_pressed()[0] and not self.click:
            self.clicked = not self.clicked
            self.click = True
        if not pg.mouse.get_pressed()[0]:
            self.click = False

    def draw(self, surf):
        if self.clicked:
            pg.draw.rect(surf, (200, 200, 200), self.rect)
            pg.draw.rect(surf, BLACK, self.rect, 4)
            draw_text(surf, self.text_on, my_font, 30, BLACK, self.position.x, self.position.y, align="center")
        else:
            pg.draw.rect(surf, WHITE, self.rect)
            pg.draw.rect(surf, BLACK, self.rect, 4)
            draw_text(surf, self.text_off, my_font, 30, BLACK, self.position.x, self.position.y, align="center")


class Connection:
    def __init__(self, neuron1, neuron2):
        self.weight = random.uniform(-1, 1)
        self.neuron1 = neuron1
        self.neuron1.set_weight(self.weight)
        self.neuron2 = neuron2
        self.neuron2.set_weight(self.weight)

    def nudge_weight(self):
        pass

    def draw(self, surf):
        pg.draw.line(surf, WHITE, self.neuron1.pos, self.neuron2.pos)

    def draw_weights(self, surf):
        dir_vec = self.neuron2.pos - self.neuron1.pos
        dir_length = dir_vec.length() * 0.35
        dir_vec.scale_to_length(dir_length)
        dir_vec = self.neuron1.pos.copy() + dir_vec + vec(0, -10)
        draw_text(surf, f'{self.weight:.2f}', my_font, 25, BLACK, dir_vec.x, dir_vec.y, align="center")


class Layer:
    def __init__(self, neurons_num, pos):
        self.pos = vec(pos)
        self.neurons = []
        self.bias = random.uniform(-1, 1)
        for i in range(neurons_num):
            self.neurons.append(Neuron(self.bias, (self.pos.x, self.pos.y + i * OFFSET_Y), RADIUS, self))

    def get_mean_pos(self):
        mean = vec(0, 0)
        for n in self.neurons:
            mean += n.pos.copy()
        mean /= len(self.neurons)
        return mean

    def update(self):
        for n in self.neurons:
            n.update()

    def draw(self, surf, mode):
        # pos_bias = self.neurons[-1].pos.copy() + vec(0, 100)
        # pg.draw.line(surf, BLACK, self.neurons[-1].pos.copy(), pos_bias)
        # pg.draw.ellipse(surf, WHITE, (pos_bias.x - 50, pos_bias.y - 25, 100, 50))
        # pg.draw.ellipse(surf, BLACK, (pos_bias.x - 50, pos_bias.y - 25, 100, 50), 3)
        # draw_text(surf, f'BIAS:{self.bias:.2f}', my_font, 30, BLACK, pos_bias.x, pos_bias.y, align="center")

        for n in range(len(self.neurons) - 1):
            pg.draw.line(surf, BLACK, self.neurons[n].pos, self.neurons[n + 1].pos)
        for n in self.neurons:
            if mode:
                n.draw_error(surf)
            else:
                n.draw(surf)


class Neural_Network:
    def __init__(self, layer_num, layers_dimensions, pos, input_num=0):
        self.pos = vec(pos)
        self.layers_dimensions = layers_dimensions
        self.connections = []
        self.layers = []
        self.inputs = []
        self.outputs_right = []

        # create layers
        for n in range(layer_num):
            self.layers.append(Layer(self.layers_dimensions[n], (self.pos.x + n * OFFSET_X, self.pos.y)))

        # create inputs
        if input_num == 0:
            for n in self.layers[0].neurons:
                self.inputs.append(Input(n.pos.copy() - vec(OFFSET_X * 0.75, 0), 20))
        else:
            for i in range(input_num):
                self.inputs.append(Input(self.layers[0].neurons[0].pos - vec(OFFSET_X * 0.75, -i * OFFSET_Y), 20))

            mean_inps = vec(0, 0)
            for i in self.inputs:
                mean_inps += i.pos.copy()
            mean_inps /= len(self.inputs)

            inps_pos_dif = self.layers[0].get_mean_pos() - mean_inps
            for i in self.inputs:
                i.offset((0, inps_pos_dif.y))

        # budge layers
        for i in range(len(self.layers) - 1):
            mean1 = vec(0, 0)
            for n in self.layers[i].neurons:
                mean1 += n.pos
            mean1 /= len(self.layers[i].neurons)

            mean2 = vec(0, 0)
            for n2 in self.layers[i + 1].neurons:
                mean2 += n2.pos
            mean2 /= len(self.layers[i + 1].neurons)

            mean_diff = mean1 - mean2
            for n2 in self.layers[i + 1].neurons:
                n2.offset((0, mean_diff.y))

        self.weights_button = Button(self.layers[0].neurons[-1].pos.copy() + vec(-150, 100),"Wgts:on", "Wgts:off")
        self.train_button = Button(self.layers[0].neurons[-1].pos.copy() + vec(-150, 150),"train", "train")
        self.error_mode_button = Button(self.layers[0].neurons[-1].pos.copy() + vec(-150, 200),"Val_Mode", "Error_Mode")

        # create right_outputs
        for n in self.layers[-1].neurons:
            self.outputs_right.append(Input(n.pos.copy() + vec(OFFSET_X * 0.75, 0), 20))

        # create connections
        for i in range(len(self.layers) - 1, 0, -1):
            for n in self.layers[i].neurons:
                for n2 in self.layers[i - 1].neurons:
                    n.add_neuron_before(n2)
                    connection = Connection(n, n2)
                    n.connections.append(connection)
                    n2.connection_to_neuron_after.append(connection)
                    self.connections.append(connection)

        # with inputs too
        for n in self.layers[0].neurons:
            for i in self.inputs:
                n.add_neuron_before(i)
                connection = Connection(n, i)
                n.connections.append(connection)
                i.connection_to_neuron_after.append(connection)
                self.connections.append(connection)

    def update(self):
        # backpropagate the error between last layer and output
        for i in range(len(self.outputs_right)):
            val = self.outputs_right[i].val -  self.layers[-1].neurons[i].val
            self.layers[-1].neurons[i].errors_accumulated[self.outputs_right[i]] = val

        self.weights_button.update()
        self.train_button.update()
        self.error_mode_button.update()

        if self.train_button.clicked:
            # for i in self.inputs:
            #     i.adjust_connection_weight()
            for l in self.layers:
                for n in l.neurons:
                    n.adjust_connection_weight()

        for i in self.inputs:
            i.update()

        for l in self.layers:
            l.update()

        for o in self.outputs_right:
            o.update()



    def draw(self, surf):
        for c in self.connections:
            c.draw(surf)
            if self.weights_button.clicked:
                c.draw_weights(surf)

        for l in self.layers:
            l.draw(surf, self.error_mode_button.clicked)

        for i in self.inputs:
            i.draw(surf)

        for o in self.outputs_right:
            o.draw(surf)

        self.weights_button.draw(surf)
        self.train_button.draw(surf)
        self.error_mode_button.draw(surf)
        # show output error
        error_place = self.outputs_right[0].pos.copy() - vec(-100, 50)
        draw_text(surf, f'Output_error(s)', my_font, 30, YELLOW, error_place.x, error_place.y, align="center")
        for i in range(len(self.outputs_right)):
            val = self.outputs_right[i].val -  self.layers[-1].neurons[i].val
            pos = self.outputs_right[i].pos.copy() + vec(100, 0)
            draw_text(surf, f'{val:.2f}', my_font, 30, YELLOW, pos.x, pos.y, align="center")
