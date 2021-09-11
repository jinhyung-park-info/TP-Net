import pygame
from Box2D.b2 import *
from pygame.locals import *
from common.Utils import *
from box2d_simulator.softbody_factory import *
from box2d_simulator.wall_factory import *
from common.Constants import *
import time
from tqdm import tqdm


# Simulation Related
class ContactListener(b2ContactListener):

    def __init__(self, world):
        b2ContactListener.__init__(self)
        self.world = world

    def BeginContact(self, contact):
        print("BeginContact")

    def EndContact(self, contact):
        print("EndContact")


def draw_polygon(polygon, body, fixture, body_number):
    vertices = [(body.transform * v) * PPM for v in polygon.vertices]
    vertices = [(v[0], SCREEN_HT - v[1]) for v in vertices]
    pygame.draw.polygon(screen, body_number_to_color[body_number], vertices)


def draw_circle(circle, body, fixture, body_number):
    position = body.transform * circle.pos * PPM
    position = (position[0], SCREEN_HT - position[1])
    pygame.draw.circle(screen, body_number_to_color[body_number], [int(x) for x in position], int(circle.radius * PPM))


if __name__ == '__main__':

    circleShape.draw = draw_circle
    polygonShape.draw = draw_polygon

    animations = []
    for force in FORCE_LST:
        for angle in ANGLE_LST:
            for init_x_pos, init_y_pos in POS_LST:
                animations.append((force, angle, init_x_pos, init_y_pos))

    animations += SIM_DATA_EVAL_CASES

    for animation_info in tqdm(animations):

        force = animation_info[0]
        angle = animation_info[1]
        init_x_pos = animation_info[2]
        init_y_pos = animation_info[3]

        # Screen Setup
        screen = pygame.display.set_mode((SCREEN_WD, SCREEN_HT), 0, 32)
        pygame.display.set_caption("Where to Go")
        clock = pygame.time.Clock()
        contactListener = ContactListener(world)
        world.contactListener = contactListener
        world = b2World(gravity=GRAVITY, doSleep=False)
        time.sleep(1)

        # Set up Simulation Environments
        init_angle = angle * math.pi / 180
        init_force = (force * math.cos(init_angle), force * math.sin(init_angle))

        # Create Walls and Softbody
        WallFactory(world, SCREEN_LENGTH, SCREEN_HEIGHT, WALL_WIDTH).create_walls()
        soft_body_factory = SoftBodyFactory(world=world,
                                            pos_x=init_x_pos,
                                            pos_y=init_y_pos,
                                            radius=RADIUS,
                                            satellite_radius=SATELLITE_RADIUS,
                                            num_satellites=NUM_PARTICLES,
                                            frequency=FREQUENCY,
                                            damping_ratio=DAMPING_RATIO)

        soft_body_center, soft_body_satellites = soft_body_factory.create_soft_body()

        for i in range(NUM_PARTICLES):
            soft_body_satellites[i].ApplyLinearImpulse(impulse=init_force, point=soft_body_center, wake=True)

        # Start the Simulation and Collect Data
        running = True
        curr_timestep = 0
        satellite_position_arrays = []

        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                    continue
                if event.type == KEYDOWN and event.key == K_ESCAPE:
                    running = False
                    continue

            screen.fill((0, 0, 0, 0))

            for body_number, body in enumerate(world.bodies):
                for fixture in body.fixtures:
                    fixture.shape.draw(body, fixture, body_number)

            world.Step(timeStep, velIters, posIters)
            world.ClearForces()
            pygame.display.flip()
            clock.tick(TARGET_FPS)

            # Save 2D Point Set data
            if DATA_SAVE_MODE:

                if curr_timestep < NUM_SEQUENCE_PER_ANIMATION:
                    satellite_pos_array = soft_body_factory.get_satellite_position_array()
                    satellite_position_arrays.append(satellite_pos_array)
                    curr_timestep += 1

                else:
                    # Stop
                    point_set_savepath = create_directory(os.path.join(SIM_DATA_PATH, 'pointset', f'force_{force}', f'angle_{angle}', f'pos_{init_x_pos}_{init_y_pos}'))
                    data_file = write_json(satellite_position_arrays, f'{point_set_savepath}/ordered_unnormalized_pointset.json')
                    data_file.close()
                    break
            else:
                curr_timestep += 1
                if curr_timestep > NUM_SEQUENCE_PER_ANIMATION:
                    break

        pygame.quit()

    print("done")
