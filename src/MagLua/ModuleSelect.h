#ifndef MODULESELECT_H
#define MODULESELECT_H

#include <QDialog>
#include <QtGui>


namespace Ui {
    class ModuleSelect;
}

class ModuleSelect : public QDialog
{
    Q_OBJECT

public:
    explicit ModuleSelect(QWidget *parent = 0);
    ~ModuleSelect();

	QStringList mods;

public slots:
	void add();
	void remove();

private:
    Ui::ModuleSelect *ui;
};

#endif // MODULESELECT_H
