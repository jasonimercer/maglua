#ifndef JOBWIDGET_H
#define JOBWIDGET_H

#include <QWidget>
#include "QMagluaHighlighter.h"
#include <QGraphicsScene>

namespace Ui {
    class JobWidget;
}

class JobWidget : public QWidget
{
    Q_OBJECT

public:
    explicit JobWidget(QWidget *parent = 0);
    ~JobWidget();

	bool loadFile(QString& filename);
	bool save();
	bool saveAs(const QString& filename="", bool askoverwrite=true);

	QString currentFile();
private:
    Ui::JobWidget *ui;
	QString filename;

	QMagLuaHighlighter* highlighter;

	QGraphicsScene scene;
};

#endif // JOBWIDGET_H
