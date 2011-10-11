#include "Group.h"
#include "DrawOpenGL.h"
#include "MainWindow.h"
#include <math.h>

DrawOpenGL::DrawOpenGL()
	: Draw()
{
	q = 0;
	refcount = 0;
	next_light = 0;
}

DrawOpenGL::~DrawOpenGL()
{
	if(q)
		gluDeleteQuadric(q);
}

void DrawOpenGL::init()
{
	q = gluNewQuadric();
}

void DrawOpenGL::reset()
{
	for(int i=0; i<next_light; i++)
	{
		glDisable(GL_LIGHT0+i);
	}

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glDepthFunc(GL_LEQUAL);
//	glEnable(GL_MULTISAMPLE);
	glEnable(GL_COLOR_MATERIAL);

	glShadeModel(GL_SMOOTH);

	glClearColor( 1,1,1, 1.0f );
	glClearDepth(1.0f);
	next_light = 0;
}

void clamp01(float& f)
{
	if(f < 0)
		f = 0;
	if(f > 1)
		f = 1;
}

void DrawOpenGL::draw(const Sphere& s, const Color& c)
{
	if(!q) init();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	const GLfloat mShininess[] = {128}; //set the shininess of the material
	const GLfloat whiteSpecularMaterial[] = {1.0, 1.0, 1.0};

	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, whiteSpecularMaterial);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mShininess);


	if(s.selected)
	{
		float cc[4];
		cc[3] = 1;
		float w = 1.0;//0.75 * cos(Singleton.time * 3.1415926);
		for(int i=0; i<3; i++)
		{
			cc[i] = c.rgba[i] + w;
			clamp01(cc[i]);
		}
		glColor4fv(cc);
	}
	else
	{
		glColor4dv(c.rgba);
	}

	glTranslatef(s.pos().x(), s.pos().y(), s.pos().z());
	gluSphere(q, s.radius(), 32, 16);
}

//void DrawOpenGL::draw(Group& group)
//{
//	Draw::draw(group);
//}

void DrawOpenGL::draw(Tube& tube)
{
	if(!q) init();
	glMatrixMode(GL_MODELVIEW);

	Vector a = tube.pos(0);
	Vector b = tube.pos(1);
	Vector c = b - a;

	float tubeHeight = c.length();
	Vector d(0,0, tubeHeight);

	float rz = Vector::radiansBetween(c, d);

	Vector n = d.cross(c);
	n.normalize();

	glPushMatrix();
	glLoadIdentity();
	const GLfloat mShininess[] = {128}; //set the shininess of the material
	const GLfloat whiteSpecularMaterial[] = {1.0, 1.0, 1.0};

	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, whiteSpecularMaterial);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mShininess);

	glColor4dv(tube.color.rgba);

	glTranslatef(a.x(), a.y(), a.z());
	glRotatef(rz*180/3.14159265358979, n.x(), n.y(), n.z());
	gluCylinder(q, tube.radius(0), tube.radius(1), tubeHeight, 32, 16);
	glPopMatrix();
}

void DrawOpenGL::draw(Light& light)
{
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0+next_light);

	float spec[4];
	float diff[4];

	for(int i=0; i<3; i++)
	{
		spec[i] = light.specular_color.component(i);
		diff[i] = light.diffuse_color.component(i);
	}
	spec[3] = 1;
	diff[3] = 1;

	glLightfv(GL_LIGHT0+next_light, GL_SPECULAR, spec);
	glLightfv(GL_LIGHT0+next_light, GL_DIFFUSE,  diff);

	GLfloat position[4];
	position[0] = light.pos().x();
	position[1] = light.pos().y();
	position[2] = light.pos().z();
	position[3] = 1.0;

	glLightfv(GL_LIGHT0+next_light, GL_POSITION, position);

	next_light++;
	glPopMatrix();
}

static void draw_gn(DrawOpenGL* d, GroupNode* gn)
{
	if(gn->n1)
	{
		draw_gn(d, gn->n1);
		draw_gn(d, gn->n2);
	}
	else
	{
		Atom* a = dynamic_cast<Atom*>(gn->v);
		if(a)
		{
			d->draw(*a, gn->v->color);
			return;
		}
		Tube* t = dynamic_cast<Tube*>(gn->v);
		if(t)
		{
			d->draw(*t);
			return;
		}
	}
}

// 	void draw(Group& group);
void DrawOpenGL::draw(Group& group)
{
	list<GroupNode*>::iterator it;

	for(it =group.precompiled.begin();
		it!=group.precompiled.end();
		it++)
	{
		draw_gn(this, *it);
	}

	if(group.root)
	{
		draw_gn(this, group.root);
	}
}

void DrawOpenGL::draw(VolumeLua& volumelua)
{
	if(!q) init();
	glMatrixMode(GL_MODELVIEW);

	glPushMatrix();
	glLoadIdentity();

	const GLfloat mShininess[] = {128}; //set the shininess of the material
	const GLfloat whiteSpecularMaterial[] = {1.0, 1.0, 1.0};
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, whiteSpecularMaterial);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mShininess);

	glColor4dv(volumelua.color.rgba);
	//glColor4f(0,0,0,1);

	vector<Tetrahedron>& t = volumelua.tetrahedron;

	glBegin(GL_TRIANGLES);
	for(unsigned int i=0; i<t.size(); i++)
	{
//		for(unsigned int f=0; f<4; f++)
		const unsigned int f = 0;
		{
			glNormal3dv(t[i].tri[f].normal.vec());

			for(int j=0; j<3; j++)
			{
				glVertex3dv(t[i].tri[f].vert[j].vec());
			}
		}
	}
	glEnd();

	glDisable(GL_LIGHTING);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glColor4f(volumelua.color.r()*0.2,volumelua.color.g()*0.2,volumelua.color.b()*0.2,1);
	glPolygonOffset(1.0, 1.0);
	glBegin(GL_LINE_LOOP);
	for(unsigned int i=0; i<t.size(); i++)
	{
//		for(unsigned int f=0; f<4; f++)
		const unsigned int f = 0;
		{
			for(int j=0; j<3; j++)
			{
				glVertex3dv(t[i].tri[f].vert[j].vec());
			}
		}
	}
	glEnd();
	glDisable(GL_POLYGON_OFFSET_FILL);
	glEnable(GL_LIGHTING);

	glPopMatrix();

}

void DrawOpenGL::draw(Camera& c)
{
	const QGLContext* con = QGLContext::currentContext();
	int windowx = con->device()->width();
	int windowy = con->device()->height();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

//	glViewport(0, 0, windowx, windowy);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	const float ratio = c.ratio;// ((float)windowx)/((float)(windowy+1));
	const float FOV = c.FOV;// 45;
	if(c.perspective)
	{
		gluPerspective(FOV, ratio, 0.1f, 10000.0f);
	}
	else
	{
		float d = c.dist();
		glOrtho(-d*0.5*ratio,d*0.5*ratio, -d*0.5, d*0.5, 0.1, 1000);
	}

	gluLookAt(c.pos.x(), c.pos.y(), c.pos.z(),
			  c.at.x(),  c.at.y(),  c.at.z(),
			  c.up.x(),  c.up.y(),  c.up.z());
}


int lua_isdrawopengl(lua_State* L, int idx)
{
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "DrawOpenGL");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}


DrawOpenGL* lua_todrawopengl(lua_State* L, int idx)
{
	DrawOpenGL** pp = (DrawOpenGL**)luaL_checkudata(L, idx, "DrawOpenGL");
	luaL_argcheck(L, pp != NULL, 1, "`DrawOpenGL' expected");
	return *pp;
}

void lua_pushdrawopengl(lua_State* L, DrawOpenGL* d)
{
	DrawOpenGL** pp = (DrawOpenGL**)lua_newuserdata(L, sizeof(DrawOpenGL**));
	*pp = d;
	luaL_getmetatable(L, "DrawOpenGL");
	lua_setmetatable(L, -2);
	d->refcount++;
}

static int l_dgl_new(lua_State* L)
{
	lua_pushdrawopengl(L, new DrawOpenGL());
	return 1;
}

static int l_dgl_gc(lua_State* L)
{
	DrawOpenGL* d = lua_todrawopengl(L, 1);
	if(!d) return 0;

	d->refcount--;
	if(d->refcount == 0)
		delete d;
	return 0;
}

static int l_dgl_reset(lua_State* L)
{
	DrawOpenGL* d = lua_todrawopengl(L, 1);
	if(!d) return 0;

	d->reset();
	return 0;
}

static int l_dgl_draw(lua_State* L)
{
	DrawOpenGL* d = lua_todrawopengl(L, 1);
	if(!d) return 0;

	if(lua_isatom(L, 2))
	{
		Atom* a = lua_toatom(L, 2);

		d->draw(*a, a->color);
		return 0;
	}

	if(lua_iscamera(L, 2))
	{
		Camera* c = lua_tocamera(L, 2);
		d->draw(*c);
		return 0;
	}

	if(lua_isgroup(L, 2))
	{
		Group* g = lua_togroup(L, 2);
		d->draw(*g);

//		d->draw(*(lua_togroup(L, 2)));
		return 0;
	}

	if(lua_islight(L, 2))
	{
		d->draw(*lua_tolight(L, 2));
		return 0;
	}

	if(lua_istube(L, 2))
	{
		d->draw(*lua_totube(L, 2));
		return 0;
	}

	if(lua_isvolumelua(L, 2))
	{
		d->draw(*lua_tovolumelua(L, 2));
		return 0;
	}

	return luaL_error(L, "Failed to draw object");
}

void lua_registerdrawopengl(lua_State* L)
{
	static const struct luaL_reg struct_m [] = { //methods
		{"__gc",         l_dgl_gc},
		{"draw",         l_dgl_draw},
		{"reset",        l_dgl_reset},
		{NULL, NULL}
	};

	luaL_newmetatable(L, "DrawOpenGL");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, struct_m);
	lua_pop(L,1); //metatable is registered

	static const struct luaL_reg struct_f [] = {
			{"new", l_dgl_new},
			{NULL, NULL}
	};

	luaL_register(L, "DrawOpenGL", struct_f);
	lua_pop(L,1);
}

