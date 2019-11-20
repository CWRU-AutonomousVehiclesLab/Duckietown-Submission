#!/usr/bin/env python3
import cv2
import numpy as np
import tensorflow as tf

from aido_schemas import EpisodeStart, protocol_agent_duckiebot1, PWMCommands, Duckiebot1Commands, LEDSCommands, RGB, \
    wrap_direct, Context, Duckiebot1Observations, JPGImage
from ik import SteeringToWheelVelWrapper

from keras.models import load_model
from frankmodel import FrankNet


#! Global Config
expect_shape = (480, 640, 3)
convertion_wrapper = SteeringToWheelVelWrapper()


class TensorflowTemplateAgent:
    def __init__(self):

        # define observation and output shapes
        self.model = load_model("FrankNet.h5")

        self.current_image = np.zeros(expect_shape)
        self.input_image = np.zeros((100, 200, 3))
        self.to_predictor = np.expand_dims(self.input_image, axis=0)

        #! for fun
        self.led_counter = 0

    def init(self, context: Context):
        context.info('init()')

    def on_received_seed(self, data: int):
        np.random.seed(data)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        context.info(f'Starting episode "{data.episode_name}".')

    #! Image pre-processing here
    def on_received_observations(self, data: Duckiebot1Observations):
        camera: JPGImage = data.camera
        self.current_image = jpg2rgb(camera.jpg_data)
        self.input_image = self.image_resize(self.current_image, width=200)
        self.input_image = self.input_image[50:150, 0:200]
        self.input_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2YUV)
        self.to_predictor = np.expand_dims(self.input_image, axis=0)

    def image_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    #! Modification here! Return with action

    def compute_action(self, observation):
        (linear, angular) = self.model.predict(observation)
        return linear, angular

    #! Major Manipulation here Should not always change
    def on_received_get_commands(self, context: Context):
        linear, angular = self.compute_action(
            self.to_predictor)  # * Changed to custom size
        #! Inverse Kinematics
        pwm_left, pwm_right = convertion_wrapper.convert(linear, angular)
        pwm_left = float(np.clip(pwm_left, -1, +1))
        pwm_right = float(np.clip(pwm_right, -1, +1))

        #! LED Commands Sherrif Duck
        grey = RGB(0.0, 0.0, 0.0)
        red = RGB(255.0, 0.0, 0.0)
        blue = RGB(0.0, 0.0, 255.0)

        led_commands = LEDSCommands(red, grey, blue, red, blue)
        if (self.led_counter < 30):
            led_commands = LEDSCommands(grey, red, blue, red, blue)
            self.led_counter += 1
        elif (self.led_counter >= 60):
            self.led_counter = 0
            led_commands = LEDSCommands(grey, red, blue, red, blue)
        elif(self.led_counter > 30):
            led_commands = LEDSCommands(blue, red, grey, blue, red)
            self.led_counter += 1

        #! Do not modify here!
        pwm_commands = PWMCommands(motor_left=pwm_left, motor_right=pwm_right)
        commands = Duckiebot1Commands(pwm_commands, led_commands)
        context.write('commands', commands)

    def finish(self, context: Context):
        context.info('finish()')


def jpg2rgb(image_data: bytes) -> np.ndarray:
    """ Reads JPG bytes as RGB"""
    from PIL import Image
    import io
    im = Image.open(io.BytesIO(image_data))
    im = im.convert('RGB')
    data = np.array(im)
    assert data.ndim == 3
    assert data.dtype == np.uint8
    return data


def main():
    node = TensorflowTemplateAgent()
    protocol = protocol_agent_duckiebot1
    wrap_direct(node=node, protocol=protocol)


if __name__ == '__main__':
    main()
