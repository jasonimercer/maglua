#include "QLineNumberTextEdit.h"
#include "MainWindow.h"
#include <QtGui>
#include <stdio.h>
#include <iostream>

using namespace std;

#define max_run_hist 1

QLuaLineNumberTextEdit::QLuaLineNumberTextEdit(lua_State *L, QLineNumberTextEdit *w)
 : QLuaWidget(L, w)
{
	stateChangeFunc = LUA_NOREF;

	connect(w->textEdit(), SIGNAL(textChanged()), this, SLOT(textChanged()));
}

QLuaLineNumberTextEdit::~QLuaLineNumberTextEdit()
{
	if(L)
	{
		if(stateChangeFunc != LUA_NOREF)
		{
			luaL_unref(L, LUA_REGISTRYINDEX, stateChangeFunc);
		}
	}
}

void QLuaLineNumberTextEdit::setStateChangeFunc(int ref)
{
	if(!L) return;

	if(stateChangeFunc != LUA_NOREF)
	{
		luaL_unref(L, LUA_REGISTRYINDEX, stateChangeFunc);
	}

	stateChangeFunc = ref;
}

void QLuaLineNumberTextEdit::textChanged()
{
	if(L && stateChangeFunc != LUA_NOREF)
	{
		lua_rawgeti(L, LUA_REGISTRYINDEX, stateChangeFunc);

		if(lua_pcall(L, 0, 0, 0))
		{
			cerr << lua_tostring(L, -1) << endl;
			QErrorMessage* msg = new QErrorMessage(Singleton.mainWindow);
			msg->showMessage( QString(lua_tostring(L, -1)).replace("\n", "<br>") );
			lua_pop(L, lua_gettop(L));
		}
		lua_gc(L, LUA_GCCOLLECT, 0);
	}
}







QLineNumberArea::QLineNumberArea(QTextEdit *editor)
	: QWidget(editor)
{
	setTextEdit(editor);

}


void QLineNumberArea::setTextEdit(QTextEdit* edit)
{
	textEditor = edit;
	setFixedWidth(textEditor->fontMetrics().width( QString("0000")));

	connect( edit->document()->documentLayout(), SIGNAL( update(const QRectF &) ), this, SLOT( update() ) );
	connect( edit->verticalScrollBar(), SIGNAL(valueChanged(int) ), this, SLOT( update() ) );
}

void QLineNumberArea::setBoldLine(int line)
{
	boldLines.push_front(line);
	while(boldLines.size() > max_run_hist)
		boldLines.removeLast();
	//boldLine = line;
	update();
}


void QLineNumberArea::paintEvent(QPaintEvent *)
{
	QAbstractTextDocumentLayout* layout = textEditor->document()->documentLayout();
	int contentsY = textEditor->verticalScrollBar()->value();
	qreal pageBottom = contentsY + textEditor->viewport()->height();
	const QFontMetrics fm = fontMetrics();
	const int ascent = fontMetrics().ascent() + 1;
	int lineCount = 1;

	QPainter p(this);

	//	textEditor->document()->

	QTextBlock block = textEditor->document()->begin();
	for(;block.isValid(); block = block.next(), lineCount++)
	{
		const QRectF boundingRect = layout->blockBoundingRect(block);

		QPointF position = boundingRect.topLeft();
		if(position.y() + boundingRect.height() < contentsY)
			continue;
		if(position.y() > pageBottom)
			break;


		const QString txt = QString::number(lineCount);
		QFont f = p.font();
		QFont q = p.font();
		const int i = boldLines.indexOf(lineCount);
		if(i!=-1)
		{
			//float r = ((float)i) / ((float)max_run_hist);
			f.setBold(true);
			p.setFont(f);
		}
		p.drawText(width() - fm.width(txt), qRound( position.y() ) - contentsY + ascent, txt);
		if(i!=-1)
		{
			p.setFont(q);
		}
	}
}





QLineNumberTextEdit::QLineNumberTextEdit(QWidget *parent) :
	QFrame(parent)
{
	//setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	setFrameStyle( QFrame::StyledPanel | QFrame::Sunken );
	setLineWidth( 2 );

	// Setup the main view
	view = new QTextEdit( this );
	view->setFontFamily( "Mono" );
	view->setLineWrapMode( QTextEdit::WidgetWidth );
	view->setFrameStyle( QFrame::NoFrame );
	view->installEventFilter( this );

	setFocusProxy(view);

	highlighter = 0;//new QLuaHilighter(document());

	connect(document(), SIGNAL(contentsChange(int,int,int)), this, SLOT(textChanged(int,int,int)) );

	// Setup the line number pane
	numbers = new QLineNumberArea(view);

	box = new QHBoxLayout( this );
	box->setSpacing( 0 );
	box->setMargin( 0 );
	box->addWidget( numbers );
	box->addWidget( view );

	numbers->setFocusProxy(view);

	setShowLineNumber(false);
	setHighlight(false);
}

void QLineNumberTextEdit::setShowLineNumber(bool b)
{
	numbers->setVisible(b);
}

void QLineNumberTextEdit::setHighlight(bool b)
{
	if(b)
	{
		if(!highlighter)
			highlighter = new QLuaHilighter(document());
	}
	else
	{
		if(highlighter)
			delete highlighter;
		highlighter = 0;
	}
}


void QLineNumberTextEdit::setBoldLine(unsigned int line)
{
	numbers->setBoldLine(line);
}

void QLineNumberTextEdit::copy()
{
	view->copy();
}

void QLineNumberTextEdit::cut()
{
	view->cut();
}

void QLineNumberTextEdit::paste()
{
	view->paste();
}

bool QLineNumberTextEdit::saveFile(const QString &fileName)
{
	QFile file(fileName);
	if (!file.open(QFile::WriteOnly | QFile::Text)) {
		QMessageBox::warning(this, tr("MagLua"),
							 tr("Cannot write file %1:\n%2.")
							 .arg(fileName)
							 .arg(file.errorString()));
		return false;
	}

	QTextStream out(&file);
	QApplication::setOverrideCursor(Qt::WaitCursor);
	out << textEdit()->toPlainText();
	QApplication::restoreOverrideCursor();

	return true;
}

void QLineNumberTextEdit::textChanged( int pos, int removed, int added )
{
	return;
	Q_UNUSED( pos );
#if 0
	if ( removed == 0 && added == 0 )
		return;

	if(doHighlighting)
	{
		QTextBlock block = highlight.block();
		QTextBlockFormat fmt = block.blockFormat();
		QColor bg = view->palette().base().color();
		fmt.setBackground( bg );
		highlight.setBlockFormat( fmt );

		int lineCount = 1;
		for ( QTextBlock block = view->document()->begin();
			 block.isValid(); block = block.next(), ++lineCount ) {

			//		if ( lineCount == infoLine ) {
			//			fmt = block.blockFormat();
			//			QColor bg = view->palette().highlight().color().light(150);
			//			fmt.setBackground( bg );

			//			highlight = QTextCursor( block );
			//			highlight.movePosition( QTextCursor::EndOfBlock, QTextCursor::KeepAnchor );
			//			highlight.setBlockFormat( fmt );

			//			break;
			//		} else if ( lineCount == warningLine ) {
			//			fmt = block.blockFormat();
			//			QColor bg = view->palette().highlight().color().light(100);
			//			fmt.setBackground( bg );

			//			highlight = QTextCursor( block );
			//			highlight.movePosition( QTextCursor::EndOfBlock, QTextCursor::KeepAnchor );
			//			highlight.setBlockFormat( fmt );

			//			break;
			//		} else if ( lineCount == errorLine ) {
			//			fmt = block.blockFormat();
			//			QColor bg = view->palette().highlight().color().light(100);
			//			fmt.setBackground( bg );

			//			highlight = QTextCursor( block );
			//			highlight.movePosition( QTextCursor::EndOfBlock, QTextCursor::KeepAnchor );
			//			highlight.setBlockFormat( fmt );

			//			break;
			//		}
		}
	}
#endif
}























int lua_istextedit(lua_State* L, int idx)
{
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "TextEdit");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}

QLuaLineNumberTextEdit* lua_toluatextedit(lua_State* L, int idx)
{
	QLuaLineNumberTextEdit** pp = (QLuaLineNumberTextEdit**)luaL_checkudata(L, idx, "TextEdit");
	luaL_argcheck(L, pp != NULL, idx, "`TextEdit' expected");
	return *pp;
}


QLineNumberTextEdit* lua_totextedit(lua_State* L, int idx)
{
	QLuaLineNumberTextEdit* tt = lua_toluatextedit(L, idx);
	if(tt)
	{
		return (QLineNumberTextEdit*)tt->widget;
	}
	return 0;
}

void lua_pushluatextedit(lua_State* L, QLuaLineNumberTextEdit* c)
{
	QLuaLineNumberTextEdit** pp = (QLuaLineNumberTextEdit**)lua_newuserdata(L, sizeof(QLuaLineNumberTextEdit**));

	*pp = c;
	luaL_getmetatable(L, "TextEdit");
	lua_setmetatable(L, -2);
	c->refcount++;
}

void lua_pushtextedit(lua_State* L, QLineNumberTextEdit* c)
{
	lua_pushluatextedit(L, new QLuaLineNumberTextEdit(L, c));
}

static int l_textedit_new(lua_State* L)
{
	lua_pushtextedit(L, new QLineNumberTextEdit);
	return 1;
}

static int l_gc(lua_State* L)
{
	QLuaLineNumberTextEdit* c = lua_toluatextedit(L, 1);
	if(!c) return 0;

	c->refcount--;
	if(c->refcount == 0)
		delete c;
	return 0;
}

static int l_tostring(lua_State* L)
{
	if(lua_istextedit(L, 1))
	{
		lua_pushstring(L, "TextEdit");
		return 1;
	}
	return 0;
}

static int l_ln(lua_State* L)
{
	QLuaLineNumberTextEdit* c = lua_toluatextedit(L, 1);
	if(!c) return 0;

	((QLineNumberTextEdit*)c->widget)->setShowLineNumber(lua_toboolean(L, 2));
	return 0;
}

static int l_hl(lua_State* L)
{
	QLuaLineNumberTextEdit* c = lua_toluatextedit(L, 1);
	if(!c) return 0;

	((QLineNumberTextEdit*)c->widget)->setHighlight(lua_toboolean(L, 2));
	return 0;
}

static int l_text(lua_State* L)
{
	QLuaLineNumberTextEdit* c = lua_toluatextedit(L, 1);
	if(!c) return 0;

	lua_pushstring(L, ((QLineNumberTextEdit*)c->widget)->textEdit()->toPlainText().toStdString().c_str());
	return 1;
}

static int l_clear(lua_State* L)
{
	QLuaLineNumberTextEdit* c = lua_toluatextedit(L, 1);
	if(!c) return 0;

	((QLineNumberTextEdit*)c->widget)->textEdit()->clear();
	return 0;
}

static int l_settext(lua_State* L)
{
	QLuaLineNumberTextEdit* c = lua_toluatextedit(L, 1);
	if(!c) return 0;

	((QLineNumberTextEdit*)c->widget)->textEdit()->clear();
	((QLineNumberTextEdit*)c->widget)->textEdit()->insertPlainText(lua_tostring(L, 2));

	return 0;
}

static int l_append(lua_State* L)
{
	QLuaLineNumberTextEdit* c = lua_toluatextedit(L, 1);
	if(!c) return 0;

	((QLineNumberTextEdit*)c->widget)->textEdit()->moveCursor(QTextCursor::End, QTextCursor::MoveAnchor);

	((QLineNumberTextEdit*)c->widget)->textEdit()->insertPlainText(lua_tostring(L, 2));

	return 0;
}

static int l_saveas(lua_State* L)
{
	QLineNumberTextEdit* c = lua_totextedit(L, 1);
	if(!c) return 0;

	lua_pushboolean(L, c->saveFile(lua_tostring(L, -1)));
	c->document()->setModified(false);
	return 1;
}

static int l_statechange(lua_State* L)
{
	QLuaLineNumberTextEdit* c = lua_toluatextedit(L, 1);
	if(!c) return 0;

	c->setStateChangeFunc(luaL_ref(L, LUA_REGISTRYINDEX));

	return 0;
}

static int l_setmodified(lua_State* L)
{
	QLineNumberTextEdit* c = lua_totextedit(L, 1);
	if(!c) return 0;

	c->textEdit()->document()->setModified(lua_toboolean(L, 2));

	return 0;
}

static int l_getmodified(lua_State* L)
{
	QLineNumberTextEdit* c = lua_totextedit(L, 1);
	if(!c) return 0;

	lua_pushboolean(L, c->textEdit()->document()->isModified());

	return 1;
}

static int l_setfontfamily(lua_State* L)
{
	QLineNumberTextEdit* c = lua_totextedit(L, 1);
	if(!c) return 0;

	c->textEdit()->setFontFamily(lua_tostring(L, 2));
	return 0;
}


void lua_registertextedit(lua_State* L)
{
	static const struct luaL_reg struct_m [] =
	{ //methods
	  {"__gc",       l_gc},
	  {"__tostring", l_tostring},
	  {"setShowLineNumber", l_ln},
	  {"setHighlight", l_hl},
	  {"text",         l_text},
	  {"clear",         l_clear},
	  {"setText",      l_settext},
	  {"append",      l_append},
	  {"saveAs",      l_saveas},
	  {"setStateChangeFunction", l_statechange},
	  {"setModified",  l_setmodified},
	  {"modified",  l_getmodified},
	  {"setFontFamily",  l_setfontfamily},
	  {NULL, NULL}
	};

	luaL_newmetatable(L, "TextEdit");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, struct_m);
	lua_pop(L,1); //metatable is registered

	static const struct luaL_reg struct_f [] = {
		{"new", l_textedit_new},
		{NULL, NULL}
	};

	luaL_register(L, "TextEdit", struct_f);
	lua_pop(L,1);
}






