#ifndef QLINENUMBERTEXTEDIT_H
#define QLINENUMBERTEXTEDIT_H

#include <QtGui>
#include "QLuaHilighter.h"
#include "QLuaWidget.h"

class QLineNumberArea;


#include <QTabWidget>
#include "QLuaWidget.h"


class QLineNumberTextEdit : public QFrame
{
    Q_OBJECT
public:
	explicit QLineNumberTextEdit(QWidget *parent = 0);

	QTextEdit* textEdit() const { return view; }
	QTextDocument* document() {return view->document();}

	void setShowLineNumber(bool b);
	void setHighlight(bool b);
	bool saveFile(const QString &fileName);

	void setBoldLine(unsigned int line);

protected:
//	void resizeEvent(QResizeEvent *event);

public slots:
	void copy();
	void cut();
	void paste();

protected slots:
	void textChanged( int pos, int added, int removed );


private slots:
//	void updateLineNumberAreaWidth();
//	void highlightCurrentLine();
//	void updateLineNumberArea(const QRect &, int);



private:
	QTextEdit* view;

	QLuaHilighter* highlighter;
	QHBoxLayout *box;
	QLineNumberArea* numbers;
	//    int infoLine;
	//    int errorLine;
	//    int warningLine;
	QTextCursor highlight;
};

class QLineNumberArea : public QWidget
{
	Q_OBJECT

public:
	QLineNumberArea(QTextEdit *editor);
	//	~LineNumberArea();

	//    void setErrorLine( int lineno );
	//    void setWarningLine( int lineno );
	//    void setInfoLine( int lineno );

	void setTextEdit(QTextEdit* edit);
	void setBoldLine(int line);
protected:
	void paintEvent( QPaintEvent *ev );
	//    bool event( QEvent *ev );

private:
	QTextEdit* textEditor;
	//    QPixmap infoMarker;
	//    QPixmap warningMarker;
	//    QPixmap errorMarker;
	int infoLine;
	int warningLine;
	int errorLine;
	QRect errorRect;
	QRect infoRect;
	QRect warningRect;
	QList<int> boldLines;
};



class QLuaLineNumberTextEdit : public QLuaWidget
{
Q_OBJECT
public:
	QLuaLineNumberTextEdit(lua_State* L, QLineNumberTextEdit* w);
	~QLuaLineNumberTextEdit();
	void setStateChangeFunc(int ref);
	int stateChangeFunc;

public slots:
	void textChanged();
};


#endif // QLINENUMBERTEXTEDIT_H

int lua_istextedit(lua_State* L, int idx);
void lua_pushtextedit(lua_State* L, QLineNumberTextEdit* c);
void lua_pushluatextedit(lua_State* L, QLuaLineNumberTextEdit* c);
void lua_registertextedit(lua_State* L);

QLuaLineNumberTextEdit* lua_toluatextedit(lua_State* L, int idx);
QLineNumberTextEdit* lua_totextedit(lua_State* L, int idx);


