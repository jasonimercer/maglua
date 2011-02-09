#include "jobwidget.h"
#include "ui_jobwidget.h"
#include <QTextEdit>
#include "QMagLuaGraphicsNode.h"

JobWidget::JobWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::JobWidget)
{
    ui->setupUi(this);


	highlighter = new QMagLuaHighlighter(ui->txtCode->textEdit()->document());

	ui->graphicsView->setScene(&scene);
	scene.addItem(new QMagLuaGraphicsNode);
}

JobWidget::~JobWidget()
{
    delete ui;
}

bool JobWidget::loadFile(QString& filename)
{

	return true;
}

QString JobWidget::currentFile()
{
	return filename;
}


bool JobWidget::save()
{
	return saveAs(filename, false);
}

bool JobWidget::saveAs(const QString& filename, bool askoverwrite)
{

	return true;
}
