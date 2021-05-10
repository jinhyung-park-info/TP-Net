from Box2D import *

DENSITY = 1.0
FRICTION = 0.0
RESTITUTION = 1.0
HARDBODYBITS = 0x0003


class HardBodyFactory:

    def __init__(self, world, posX, posY, mainRadius):
        self.world = world
        self.posX = posX
        self.posY = posY
        self.mainRadius = mainRadius
        self.mainBody = None


    def createHardBody(self):
        # Creates a hard body
        hardBodyDef = b2BodyDef()
        hardBodyDef.type = b2_dynamicBody
        hardBodyDef.position.Set(self.posX, self.posY)
        hardBody = self.world.CreateBody(hardBodyDef)
        hardBodyShape = b2CircleShape(radius = self.mainRadius)
        hardBodyFixture = b2FixtureDef(shape = hardBodyShape, density = DENSITY, friction = FRICTION, restitution = RESTITUTION)
        hardBodyFixture.filter.categoryBits = HARDBODYBITS
        hardBodyFixture.filter.maskBits = 0x0001 | HARDBODYBITS
        hardBody.CreateFixture(hardBodyFixture)
        self.mainBody = hardBody
        return self.mainBody
