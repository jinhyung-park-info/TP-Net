from Box2D import *
import math
from common.Constants import SOFTBODY_FRICTION, SOFTBODY_RESTITUTION, SOFTBODY_DENSITY, SOFTBODYBITS


class SoftBodyFactory:

    def __init__(self, world, pos_x, pos_y, satellite_radius, num_satellites, radius, frequency, damping_ratio):
        self.world = world
        self.posX = pos_x
        self.posY = pos_y
        self.radius = radius
        self.satellite_radius = satellite_radius
        self.num_satellites = num_satellites
        self.frequency = frequency
        self.damping_ratio = damping_ratio
        self.satellites = []

    def create_soft_body(self):
        self._create_satellites()
        for i in range(1, 13):
            self._create_n_apart_joints(i)
        return [(self.posX, self.posY), self.satellites]

    def get_satellite_position_array(self):
        arr = []
        for satellite in self.satellites:
            position = satellite.worldCenter
            arr.extend([round(position[0], 4), round(position[1], 4)])
        return arr

    def _get_satellite_lv_array(self):
        arr = []
        for satellite in self.satellites:
            linear_velocity = satellite.linearVelocity
            arr.append([round(linear_velocity[0], 4), round(linear_velocity[1], 4)])
        return arr

    def _get_satellite_av_array(self):
        arr = []
        for satellite in self.satellites:
            angular_velocity = satellite.angularVelocity
            arr.append(round(angular_velocity, 4))
        return arr

    def _create_satellites(self):
        for i in range(self.num_satellites):
            sat_body_def = b2BodyDef()
            sat_body_def.type = b2_dynamicBody
            angle = 0 + i * (math.pi / (self.num_satellites / 2))
            sat_body_def.position.Set(self.posX + self.radius * math.sin(angle), self.posY + self.radius * math.cos(angle))
            sat_body = self.world.CreateBody(sat_body_def)
            sat_body_shape = b2CircleShape(radius=self.satellite_radius)
            sat_body_fixture = b2FixtureDef(shape=sat_body_shape, density=SOFTBODY_DENSITY, friction=SOFTBODY_FRICTION, restitution=SOFTBODY_RESTITUTION)
            sat_body_fixture.filter.categoryBits = SOFTBODYBITS
            sat_body_fixture.filter.maskBits = 0x0001 | SOFTBODYBITS
            sat_body.CreateFixture(sat_body_fixture)
            self.satellites.append(sat_body)

    def _create_n_apart_joints(self, n):
        for i in range(self.num_satellites):
            joint_def = b2DistanceJointDef()
            joint_def.Initialize(self.satellites[i], self.satellites[(i+n) % self.num_satellites],
                                 self.satellites[i].worldCenter, self.satellites[(i+n) % self.num_satellites].worldCenter)

            joint_def.frequencyHz = self.frequency
            joint_def.dampingRatio = self.damping_ratio
            joint_def.collideConnected = False
            self.world.CreateJoint(joint_def)


