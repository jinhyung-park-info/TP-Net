from Box2D import *
from common.Constants import WALL_DENSITY, WALL_RESTITUTION, WALL_FRICTION


class WallFactory:

    def __init__(self, world, screen_length, screen_height, wall_width):
        self.world = world
        self.screen_length = screen_length
        self.screen_height = screen_height
        self.wall_width = wall_width

    def create_walls(self):
        self._create_horizontal_walls()
        self._create_vertical_walls()

    def _create_horizontal_walls(self):
        for i in range(2):
            wall_body_def = b2BodyDef()
            if i == 0:
                wall_body_def.position.Set(0, self.screen_height - self.wall_width)
            else:
                wall_body_def.position.Set(0, self.wall_width)

            wall_body = self.world.CreateBody(wall_body_def)
            wall_shape = b2PolygonShape()
            wall_shape.SetAsBox(self.screen_length, self.wall_width)
            wall_shape.restitution = WALL_RESTITUTION
            wall_body_fixture_def = b2FixtureDef()
            wall_body_fixture_def.shape = wall_shape
            wall_body_fixture_def.density = WALL_DENSITY
            wall_body_fixture_def.friction = WALL_FRICTION
            wall_body.CreateFixture(wall_body_fixture_def)

    def _create_vertical_walls(self):
        for i in range(2):
            wall_body_def = b2BodyDef()
            if i == 0:
                wall_body_def.position.Set(self.wall_width, self.screen_height)
            else:
                wall_body_def.position.Set(self.screen_length - self.wall_width, self.screen_height)
            wall_body = self.world.CreateBody(wall_body_def)
            wall_shape = b2PolygonShape()
            wall_shape.SetAsBox(self.wall_width, self.screen_length)
            wall_shape.restitution = WALL_RESTITUTION
            wall_body_fixture_def = b2FixtureDef()
            wall_body_fixture_def.shape = wall_shape
            wall_body_fixture_def.density = WALL_DENSITY
            wall_body_fixture_def.friction = WALL_FRICTION
            wall_body.CreateFixture(wall_body_fixture_def)

