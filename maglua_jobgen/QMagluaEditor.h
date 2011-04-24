#ifndef QMagLuaEditor_H
#define QMagLuaEditor_H

#include <QFrame>
#include <QPixmap>
#include <QTextCursor>

class QTextEdit;
class QHBoxLayout;

class InfoBar : public QWidget
{
	Q_OBJECT
public:
	InfoBar(QWidget *parent);
	~InfoBar();

	void setCurrentLine(int lineno);
	void setStopLine(int lineno);
	void setBugLine(int lineno);

	void setTextEdit(QTextEdit *edit);
	void paintEvent(QPaintEvent *ev);

protected:
	bool event(QEvent *ev);

private:
	QTextEdit* edit;
	QPixmap stopMarker;
	QPixmap currentMarker;
	QPixmap bugMarker;
	int stopLine;
	int currentLine;
	int bugLine;
	QRect stopRect;
	QRect currentRect;
	QRect bugRect;
};


class QMagLuaEditor : public QFrame
{
Q_OBJECT
public:
	QMagLuaEditor(QWidget *parent = 0);
	~QMagLuaEditor();

	QTextEdit *textEdit() const;

	void setCurrentLine(int lineno);
	void setStopLine(int lineno);
	void setBugLine(int lineno);

	bool eventFilter(QObject *obj, QEvent *event);

signals:
	void mouseHover(const QString &word);
	void mouseHover(const QPoint &pos, const QString &word);

protected slots:
	void textChanged(int pos, int added, int removed);



private:
	int currentLine;

	QHBoxLayout* hbox;
	InfoBar* info;
	QTextEdit* view;

	QTextCursor cursor;

};

#endif // QMagLuaEditor_H
