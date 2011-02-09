#include "qmagluaeditor.h"
#include <QtGui>

InfoBar::InfoBar(QWidget *parent)
	: QWidget(parent), edit(0), stopLine(-1), currentLine(-1), bugLine(-1)
	//	: QWidget(parent), edit(0), stopLine(3), currentLine(5), bugLine(11)
{
	setFixedWidth(fontMetrics().width(QString("0000") + 40));
	stopMarker = QPixmap("/users/cmms/jmercer/.rhel5_backup/.kde/share/icons/nuvoX_0.6/32x32/actions/no.png").scaled(16, 16);
	currentMarker = QPixmap("/users/cmms/jmercer/.rhel5_backup/.kde/share/icons/nuvoX_0.6/32x32/actions/next.png").scaled(16, 16);
	bugMarker = QPixmap("/users/cmms/jmercer/.rhel5_backup/.kde/share/icons/nuvoX_0.6/32x32/apps/bug.png").scaled(16, 16);

//	stopMarker = QPixmap("images/no.png");
//	currentMarker = QPixmap("images/next.png");
//	bugMarker = QPixmap("images/bug.png");


}

InfoBar::~InfoBar()
{
}

void InfoBar::setCurrentLine(int lineno)
{
	currentLine = lineno;
}

void InfoBar::setStopLine(int lineno)
{
	stopLine = lineno;
}

void InfoBar::setBugLine(int lineno)
{
	bugLine = lineno;
}

void InfoBar::setTextEdit(QTextEdit *edit)
{
	this->edit = edit;
	connect(edit->document()->documentLayout(), SIGNAL(update(const QRectF &)), this, SLOT(update()));
	connect(edit->verticalScrollBar(), SIGNAL(valueChanged(int)), this, SLOT(update()));
}

void InfoBar::paintEvent(QPaintEvent *)
{
	QAbstractTextDocumentLayout *layout = edit->document()->documentLayout();
	int contentsY = edit->verticalScrollBar()->value();
	qreal pageBottom = contentsY + edit->viewport()->height();
	const QFontMetrics fm = fontMetrics();
	const int ascent = fontMetrics().ascent() + 1; // height = ascent + descent + 1
	int lineCount = 1;

	QPainter p(this);

	bugRect = QRect();
	stopRect = QRect();
	currentRect = QRect();

	for(QTextBlock block = edit->document()->begin(); block.isValid(); block = block.next(), ++lineCount)
	{
		const QRectF boundingRect = layout->blockBoundingRect(block);

		QPointF position = boundingRect.topLeft();
		if(position.y() + boundingRect.height() < contentsY)
			continue;
		if(position.y() > pageBottom)
			break;

		const QString txt = QString::number(lineCount);
		p.drawText(width() - fm.width(txt), qRound(position.y()) - contentsY + ascent, txt);

		// Bug marker
		if(bugLine == lineCount)
		{
			p.drawPixmap(1, qRound(position.y()) - contentsY, bugMarker);
			bugRect = QRect(1, qRound(position.y()) - contentsY, bugMarker.width(), bugMarker.height());
		}

		// Stop marker
		if(stopLine == lineCount)
		{
			p.drawPixmap(1, qRound(position.y()) - contentsY, stopMarker);
			stopRect = QRect(1, qRound(position.y()) - contentsY, stopMarker.width(), stopMarker.height());
		}

		// Current line marker
		if(currentLine == lineCount)
		{
			p.drawPixmap(1, qRound(position.y()) - contentsY, currentMarker);
			currentRect = QRect(1, qRound(position.y()) - contentsY, currentMarker.width(), currentMarker.height());
		}
	}
}

bool InfoBar::event(QEvent *event)
{
	if(event->type() == QEvent::ToolTip)
	{
		QHelpEvent *helpEvent = static_cast<QHelpEvent *>(event);

		if(stopRect.contains(helpEvent->pos()))
		{
			QToolTip::showText(helpEvent->globalPos(), "Stop Here");
		}
		else if(currentRect.contains(helpEvent->pos()))
		{
			QToolTip::showText(helpEvent->globalPos(), "Current Line");
		}
		else if(bugRect.contains(helpEvent->pos()))
		{
			QToolTip::showText(helpEvent->globalPos(), "Error Line");
		}
	}

	return QWidget::event(event);
}






QMagLuaEditor::QMagLuaEditor(QWidget *parent)
	: QFrame(parent)
{
	setFrameStyle(QFrame::StyledPanel | QFrame::Sunken);
	setLineWidth(2);

	// Setup the main view
	view = new QTextEdit(this);
	view->setFontFamily("Courier");
	view->setLineWrapMode(QTextEdit::NoWrap);
	view->setFrameStyle(QFrame::NoFrame);
	view->installEventFilter(this);

	view->setTabStopWidth(16*4);

	connect(view->document(), SIGNAL(contentsChange(int,int,int)), this, SLOT(textChanged(int,int,int)));

	// Setup the info pane
	info = new InfoBar(this);
	info->setTextEdit(view);

	// Testing...
	info->setStopLine(3);
	info->setBugLine(10);
	setCurrentLine(10);

	hbox = new QHBoxLayout(this);
	hbox->setSpacing(0);
	hbox->setMargin(0);
	hbox->addWidget(info);
	hbox->addWidget(view);

	QFile file("../examples/databaseExamples/code.lua");
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
		return;

	while(!file.atEnd())
	{
		view->insertPlainText(file.readLine());
	}
}

QMagLuaEditor::~QMagLuaEditor()
{
}

void QMagLuaEditor::setCurrentLine(int lineno)
{
	currentLine = lineno;
	info->setCurrentLine(lineno);
	textChanged(0, 0, 1);
}

void QMagLuaEditor::setStopLine(int lineno)
{
	info->setStopLine(lineno);
}

void QMagLuaEditor::setBugLine(int lineno)
{
	info->setBugLine(lineno);
}

void QMagLuaEditor::textChanged(int pos, int removed, int added)
{
	Q_UNUSED(pos);

	if(removed == 0 && added == 0)
		return;

	QTextBlock block = cursor.block();
	QTextBlockFormat fmt = block.blockFormat();
	QColor bg = view->palette().base().color();
	fmt.setBackground(bg);
	cursor.setBlockFormat(fmt);

	int lineCount = 1;
	for (QTextBlock block = view->document()->begin();
	block.isValid(); block = block.next(), ++lineCount)
	{
		if(lineCount == currentLine)
		{
			fmt = block.blockFormat();
			QColor bg = view->palette().highlight().color().light(175);
			fmt.setBackground(bg);

			cursor = QTextCursor(block);
			cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::KeepAnchor);
			cursor.setBlockFormat(fmt);

			break;
		}
	}
}

QTextEdit* QMagLuaEditor::textEdit() const
{
	return view;
}


bool QMagLuaEditor::eventFilter(QObject *obj, QEvent *event)
{
	if(obj != view)
		return QFrame::eventFilter(obj, event);

	if(event->type() == QEvent::ToolTip)
	{
		QHelpEvent *helpEvent = static_cast<QHelpEvent *>(event);

		QTextCursor cursor = view->cursorForPosition(helpEvent->pos());
		cursor.movePosition(QTextCursor::StartOfWord, QTextCursor::MoveAnchor);
		cursor.movePosition(QTextCursor::EndOfWord, QTextCursor::KeepAnchor);

		QString word = cursor.selectedText();
		emit mouseHover(word);
		emit mouseHover(helpEvent->pos(), word);

		//QToolTip::showText(helpEvent->globalPos(), word); // For testing
	}
	return false;
}
