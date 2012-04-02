#include "Group.h"
#include "DrawOpenGL.h"
#include "Atom.h"
#include "Camera.h"

#include "Transformation.h"
#include "TransformationScale.h"
#include "TransformationRotate.h"
#include "TransformationTranslate.h"

#include "luagraphics_qtgl_global.h"
#include "info.h"

using namespace std;


//#define DD 	printf("(%s:%i)\n", __FILE__, __LINE__); fflush(stdout);
#define DD

DrawOpenGL::DrawOpenGL()
    : Draw(hash32("DrawOpenGL"))
{
DD;
	   q = 0;
	next_light = 0;
}

DrawOpenGL::~DrawOpenGL()
{
    if(q)
        gluDeleteQuadric(q);
}

int DrawOpenGL::luaInit(lua_State*)
{
	DD;
	return 0;
}

void DrawOpenGL::push(lua_State* L)
{
	DD;
	luaT_push<DrawOpenGL>(L, this);
}

void DrawOpenGL::init()
{
	DD;
	q = gluNewQuadric();
	//printf("q %p\n", q);
}

void DrawOpenGL::reset()
{
    for(int i=0; i<next_light; i++)
    {
        glDisable(GL_LIGHT0+i);
    }
	DD;

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glDepthFunc(GL_LEQUAL);
    //	glEnable(GL_MULTISAMPLE);
    glEnable(GL_COLOR_MATERIAL);
	DD;

    glShadeModel(GL_SMOOTH);

	//glClearColor( 1,1,1, 1.0f );
	//glClearDepth(1.0f);
    next_light = 0;
}


void DrawOpenGL::draw(Sphere& s)
{
	DD;
	if(!q) init();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    const GLfloat mShininess[] = {128}; //set the shininess of the material
    const GLfloat whiteSpecularMaterial[] = {1.0, 1.0, 1.0};

    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, whiteSpecularMaterial);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mShininess);


	DD;
	glColor4dv(s.color->rgba);

    glTranslatef(s.pos()->x(), s.pos()->y(), s.pos()->z());
    gluSphere(q, s.radius(), 32, 16);
	DD;
}


void DrawOpenGL::draw(Tube& tube)
{
    if(!q) init();
    glMatrixMode(GL_MODELVIEW);

	DD;
	Vector a = tube.pos(0);
    Vector b = tube.pos(1);
    Vector c = b - a;
	DD;

    float tubeHeight = c.length();
    Vector d(0,0, tubeHeight);

    float rz = Vector::radiansBetween(c, d);

    Vector n = d.cross(c);
    n.normalize();

    glPushMatrix();
	//glLoadIdentity();
    const GLfloat mShininess[] = {128}; //set the shininess of the material
    const GLfloat whiteSpecularMaterial[] = {1.0, 1.0, 1.0};

//    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, whiteSpecularMaterial);
//    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mShininess);

    glColor4dv(tube.color->rgba);

    glTranslatef(a.x(), a.y(), a.z());
    glRotatef(rz*180.0/3.14159265358979, n.x(), n.y(), n.z());
    gluCylinder(q, tube.radius(0), tube.radius(1), tubeHeight, 32, 16);

	gluQuadricOrientation(q, GLU_INSIDE);
	gluDisk(q, 0, tube.radius(0), 32, 2);
	gluQuadricOrientation(q, GLU_OUTSIDE);
	//gluDisk(q, 0, tube.radius(1), 32, 2);
	glPopMatrix();
	DD;
}

void DrawOpenGL::draw(Light& light)
{
	DD;
	glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
	//glLoadIdentity();

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0+next_light);

	float s[4] = {1,1,1,1};
	float d[4] = {0.5,0.5,0.5,1};

//    for(int i=0; i<3; i++)
//    {
//		s[i] = light.specular_color->rgba[i]*0.01;
//		d[i] = light.diffuse_color->rgba[i]*0.01;
//    }
//	s[3] = 1;
//	d[3] = 1;

    glLightfv(GL_LIGHT0+next_light, GL_SPECULAR, s);
    glLightfv(GL_LIGHT0+next_light, GL_DIFFUSE,  d);

    GLfloat position[4];
    position[0] = light.pos()->x();
    position[1] = light.pos()->y();
    position[2] = light.pos()->z();
    position[3] = 1.0;

    glLightfv(GL_LIGHT0+next_light, GL_POSITION, position);

    next_light++;
    glPopMatrix();
	DD;
}

void DrawOpenGL::draw(Camera& c)
{
    //const QGLContext* con = QGLContext::currentContext();
    //int windowx = con->device()->width();
    //int windowy = con->device()->height();

	DD;
	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const float ratio = c.ratio;// ((float)windowx)/((float)(windowy+1));
    const float FOV = c.FOV;// 45;
	if(c.perspective || 1)
    {
        gluPerspective(FOV, ratio, 0.1f, 10000.0f);
    }
    else
    {
        float d = c.dist();
        glOrtho(-d*0.5*ratio,d*0.5*ratio, -d*0.5, d*0.5, 0.1, 1000);
    }

    gluLookAt(c.pos->x(), c.pos->y(), c.pos->z(),
              c.at->x(),  c.at->y(),  c.at->z(),
              c.up->x(),  c.up->y(),  c.up->z());
	DD;
}

void DrawOpenGL::inner_draw(Transformation& t)
{
	DD;
	{
		Scale* tt = dynamic_cast<Scale*>(&t);
		if(tt)
		{
			glScalef(t.values[0], t.values[1], t.values[2]);
		}
	}
	{
		Rotate* tt = dynamic_cast<Rotate*>(&t);
		if(tt)
		{
			glRotatef(t.values[0], 1, 0, 0);
			glRotatef(t.values[1], 0, 1, 0);
			glRotatef(t.values[2], 0, 0, 1);
		}
	}

	DD;
	{
		Translate* tt = dynamic_cast<Translate*>(&t);
		if(tt)
		{
			glTranslatef(t.values[0], t.values[1], t.values[2]);
		}
	}
	DD;

	for(unsigned int i=0; i<t.transformations.size(); i++)
	{
		if(  t.transformations[i] )
		{
			glPushMatrix();
			//printf("%i %p\n", i, t.transformations[i]);
			inner_draw(* t.transformations[i]);
			glPopMatrix();
		}
	}

	DD;
	for(unsigned int i=0; i<t.volumes.size(); i++)
	{
		DD;
		Volume* v = t.volumes[i];
		{
			DD;
			Tube* vv = dynamic_cast<Tube*>(v);
			if(vv)
			{
				DD;
				draw(*vv);
				DD;
				continue;
			}
		}
		DD;
		{
			Sphere* vv = dynamic_cast<Sphere*>(v);
			if(vv)
			{
				draw(*vv);
				continue;
			}
		}
		DD;
		{
			Light* vv = dynamic_cast<Light*>(v);
			if(vv)
			{
				draw(*vv);
				continue;
			}
		}
		DD;
	}

	DD;

}


void DrawOpenGL::draw(Transformation& t)
{
	glPushMatrix();
	inner_draw(t);
	glPopMatrix();
}


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* DrawOpenGL::luaMethods()
{
    if(m[127].name)return m;

    merge_luaL_Reg(m, Draw::luaMethods());
    static const luaL_Reg _m[] =
    {
        {NULL, NULL}
    };
    merge_luaL_Reg(m, _m);
    m[127].name = (char*)1;
    return m;
}







extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>

LUAGRAPHICS_QTGLSHARED_EXPORT int lib_register(lua_State* L);
LUAGRAPHICS_QTGLSHARED_EXPORT int lib_version(lua_State* L);
LUAGRAPHICS_QTGLSHARED_EXPORT const char* lib_name(lua_State* L);
LUAGRAPHICS_QTGLSHARED_EXPORT int lib_main(lua_State* L);
}

#include <stdio.h>
#include "QGraphicsItemLua.h"
#include "QGraphicsSceneLua.h"
#include "QTextEditItemLua.h"
#include "QPushButtonItemLua.h"
#include "QSliderItemLua.h"
LUAGRAPHICS_QTGLSHARED_EXPORT int lib_register(lua_State* L)
{
	luaT_register<DrawOpenGL>(L);
	luaT_register<QGraphicsItemLua>(L);
	luaT_register<QGraphicsSceneLua>(L);
	luaT_register<QTextEditItemLua>(L);
	luaT_register<QPushButtonItemLua>(L);
	luaT_register<QSliderItemLua>(L);
	return 0;
}

LUAGRAPHICS_QTGLSHARED_EXPORT int lib_version(lua_State* L)
{
    return __revi;
}

const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
    return "LuaGraphicsQTGL";
#else
    return "LuaGraphicsQTGL-Debug";
#endif
}

int lib_main(lua_State* L)
{
    return 0;
}
