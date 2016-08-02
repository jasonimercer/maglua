#include "ModuleSelect.h"
#include "ui_ModuleSelect.h"

#include <QSettings>
#include <QtGui>
#include <QFileDialog>

ModuleSelect::ModuleSelect(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ModuleSelect)
{
    ui->setupUi(this);

	QSettings settings("Mercer", "MagLuaFrontend");
	mods = settings.value("modules", QStringList()).toStringList();


	mods.sort();
	mods.removeDuplicates();

	ui->list->addItems(mods);

	connect(ui->cmdAdd, SIGNAL(pressed()), this, SLOT(add()));
	connect(ui->cmdRemove, SIGNAL(pressed()), this, SLOT(remove()));
}

void ModuleSelect::add()
{
	QStringList filenames = QFileDialog::getOpenFileNames(this, tr("Select Module"), "", tr("Shared Objects (*.so)"));

	if(filenames.size())
	{
		mods << filenames;
		mods.sort();
		mods.removeDuplicates();
		ui->list->clear();
		ui->list->addItems(mods);
	}
}

void ModuleSelect::remove()
{
	const QList<QListWidgetItem *> ii = ui->list->selectedItems();
	for(int i=0; i<ii.length(); i++)
	{
		mods.removeOne(ii.at(i)->text());
	}
	ui->list->clear();
	ui->list->addItems(mods);
}


ModuleSelect::~ModuleSelect()
{

    delete ui;
}
