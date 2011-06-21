#ifndef QLINENUMBERTEXTEDIT_H
#define QLINENUMBERTEXTEDIT_H

#include <QtGui>
#include "QLuaHilighter.h"

class QLineNumberArea;

class QLineNumberTextEdit : public QFrame
{
    Q_OBJECT
public:
	explicit QLineNumberTextEdit(QWidget *parent = 0);

	QTextEdit* textEdit() const { return view; }
	QTextDocument* document() {return view->document();}

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
};



#endif // QLINENUMBERTEXTEDIT_H
