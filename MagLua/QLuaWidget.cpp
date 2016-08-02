#include "QLuaWidget.h"
#include "QLuaAction.h"
#include "MainWindow.h"

QLuaWidget::QLuaWidget(lua_State* _L, QWidget* _widget)
	: L(_L), widget(_widget), refcount(0)
{

}

QLuaWidget::~QLuaWidget()
{
	if(widget)
		delete widget;

	while(children.size())
		removeChild(children.first());

	while(children_actions.size())
		removeChild(children_actions.first());
}

// manages refcounts
void QLuaWidget::addChild(QLuaWidget* w)
{
	w->refcount++;
	children.push_back(w);
}

void QLuaWidget::removeChild(QLuaWidget* w)
{
	for(int i=0; i<children.size(); i++)
	{
		if(children.at(i) == w)
		{
			children.at(i)->refcount--;
			if(children.at(i)->refcount == 0)
			{
				delete children.at(i);
			}
			children.removeAt(i);
			return;
		}
	}
}

// manages refcounts
void QLuaWidget::addChild(QLuaAction* w)
{
	w->refcount++;
	children_actions.push_back(w);
}

void QLuaWidget::removeChild(QLuaAction* w)
{
	for(int i=0; i<children_actions.size(); i++)
	{
		if(children_actions.at(i) == w)
		{
			children_actions.at(i)->refcount--;
			if(children_actions.at(i)->refcount == 0)
			{
				delete children_actions.at(i);
			}
			children_actions.removeAt(i);
			return;
		}
	}
}

void QLuaWidget::setFocus()
{
	widget->setFocus();
}
