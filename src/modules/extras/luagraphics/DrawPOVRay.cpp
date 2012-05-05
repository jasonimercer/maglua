#include "Group.h"
#include "DrawPOVRay.h"
#include "Atom.h"
#include "Camera.h"
#include <string.h>
#include <fstream>
#include <list>

#include "Transformation.h"
#include "TransformationScale.h"
#include "TransformationRotate.h"
#include "TransformationTranslate.h"


using namespace std;

DrawPOVRay::DrawPOVRay()
	: Draw(hash32("DrawPOVRay"))
{
}

DrawPOVRay::~DrawPOVRay()
{
	out.close();
}


void DrawPOVRay::setFileName(const char* filename)
{
	out.open(filename);
	init();
}

void DrawPOVRay::close()
{
	out.flush();
	out.close();
}



int DrawPOVRay::luaInit(lua_State* L)
{
	if(lua_isstring(L, 1))
	{
		setFileName(lua_tostring(L, 1));
	}
	return 0;
}

void DrawPOVRay::push(lua_State* L)
{
	luaT_push<DrawPOVRay>(L, this);
}
	

void DrawPOVRay::init()
{
	out << "// you will need to explicitly disable Vista buffers when rendering this" << endl;
	out << "// povray -UV file.pov" << endl;
	out << "background {color rgb <1,1,1>}" << endl;
	out << "global_settings { assumed_gamma 2.2 ambient_light rgb < 1, 1, 1 > }" << endl;
	
	// macro defs
	out << 
	"#macro set_solid_material( mcolor )\n"
	"no_shadow\n"
	"texture {\n"
	"	pigment { mcolor }\n"
	"	finish {\n"
	"		specular 0.2\n"
	"		roughness 0.02\n"
	"	}\n"
	"}\n"
	"#end\n" << endl;
	
	out << 
	"#macro draw_solid_tube(pos1, rad1, pos2, rad2, tcolor)\n"
	"cone\n"
	"{\n"
    "	pos1, rad1, pos2, rad2\n"
	"	set_solid_material( tcolor )\n"
	"}\n"
	"#end\n" << endl;
	
	out << 
	"#macro draw_solid_sphere( pos, rad, scolor )\n"
	"sphere {\n"
	"	pos, rad\n"
	"	set_solid_material( scolor )\n"
	"}\n"
	"#end\n" << endl;
	
	out << 
	"#macro make_light_source( pos, dcolor, scolor )\n"
	"light_source {\n"
	"	pos\n"
	"	dcolor\n"
	"	parallel\n"
	"	shadowless\n"
	"}\n"
	"light_source {\n"
	"	pos\n"
	"	scolor\n"
	"	parallel\n"
	"}\n"
	"#end\n" << endl;
	
	out << 
	"#macro draw_solid_tri( v1, v2, v3, scolor )\n"
	"triangle {\n"
	"	v1, v2, v3\n"
	"	set_solid_material( scolor )\n"
	"}\n"
	"#end\n" << endl;
}

/*
void DrawPOVRay::draw(VolumeLua& volumelua)
{
	vector<Tetrahedron>& t = volumelua.tetrahedron;
	for(unsigned int i=0; i<t.size(); i++)
	{
		out << "draw_solid_tri( " << 
		"<" << t[i].tri[0].vert[0] << ">, " <<
		"<" << t[i].tri[0].vert[1] << ">, " << 
		"<" << t[i].tri[0].vert[2] << ">, " <<
		"rgbt<" << volumelua.color.toVector() << ", " << volumelua.color.t() << ">)" << endl;
	}
}
*/




void DrawPOVRay::draw(Sphere& s)
{
	out << "draw_solid_sphere( <" << *s.pos() << ">, "
		<< s.radius() << ", rgbt<" 
		<< s.color->toVector() << ", " << s.color->t() << ">)" << endl;
}

void DrawPOVRay::draw(Tube& t)
{
	out << "draw_solid_tube( <" << t.pos(0) << ">, " << t.radius(0) << ", <"
		<< t.pos(1) << ">, " << t.radius(1) << ", rgbt<"
		<< t.color->toVector() << ", " << t.color->t() << ">)" << endl;
}
/*
static void draw_gn(DrawPOVRay* d, GroupNode* gn)
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
}*/
	
// 	void draw(Group& group);
void DrawPOVRay::draw(Group& group)
{
	/*
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
	}*/
}

void DrawPOVRay::draw(Light& light)
{
	out <<
	"make_light_source( <" << *light.pos() << ">, "
	"rgb <" << light.diffuse_color->toVector() << ">, "
	"rgb <" << light.specular_color->toVector() << ">) " << endl;
}

void DrawPOVRay::draw(Camera& c)
{
	out << "camera {" << endl;
// 	out << "  look_at <" << c.at << ">" << endl;
	out << "  perspective" << endl;
	out << "  location <" << *c.pos << ">" << endl;
	out << "  direction <" << *c.forward << ">" << endl;
	out << "  right <" << 1.33* *c.right << ">" << endl;
	out << "  up <" << *c.up << ">" << endl;
	out << "}" << endl;
}

void DrawPOVRay::draw(Transformation& t)
{
// 	if(t.volumes.size() < 2)
// 		out << "merge {" << endl;
// 	else
 	out << "union {" << endl;

	for(unsigned int i=0; i<t.volumes.size(); i++)
	{
		Volume* v = t.volumes[i];
		{
			Tube* vv = dynamic_cast<Tube*>(v);
			if(vv)
			{
				draw(*vv);
				continue;
			}
		}
		{
			Sphere* vv = dynamic_cast<Sphere*>(v);
			if(vv)
			{
				draw(*vv);
				continue;
			}
		}

		{
			Light* vv = dynamic_cast<Light*>(v);
			if(vv)
			{
				draw(*vv);
				continue;
			}
		}
	}

	for(unsigned int i=0; i<t.transformations.size(); i++)
	{
		draw(* t.transformations[i]);
	}
	
	{
		Scale* tt = dynamic_cast<Scale*>(&t);
		if(tt)
		{
			out << "scale <" << t.values[0] << ", " << t.values[1] << ", " << t.values[2] << ">" << endl;
		}
	}
	{
		Rotate* tt = dynamic_cast<Rotate*>(&t);
		if(tt)
		{
			out << "rotate <" << t.values[0] << ", 0, 0>" << endl;
			out << "rotate <0, " << t.values[1] << ", 0>" << endl;
			out << "rotate <0, 0, " << t.values[2] << ">" << endl;		}
	}
	{
		Translate* tt = dynamic_cast<Translate*>(&t);
		if(tt)
		{
			out << "translate <" << t.values[0] << ", " << t.values[1] << ", " << t.values[2] << ">" << endl;	
		}
	}
	
	out << "}" << endl;
}

static int l_close(lua_State* L)
{
	LUA_PREAMBLE(DrawPOVRay, d, 1);
	d->close();
	return 0;
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* DrawPOVRay::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, Draw::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"close", l_close},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}
