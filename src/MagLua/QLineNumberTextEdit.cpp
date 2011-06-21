#include "QLineNumberTextEdit.h"
#include <QtGui>

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
		p.drawText(width() - fm.width(txt), qRound( position.y() ) - contentsY + ascent, txt );
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

	highlighter = new QLuaHilighter(document());

	connect(document(), SIGNAL(contentsChange(int,int,int)), this, SLOT(textChanged(int,int,int)) );

	// Setup the line number pane
	numbers = new QLineNumberArea(view);

	box = new QHBoxLayout( this );
	box->setSpacing( 0 );
	box->setMargin( 0 );
	box->addWidget( numbers );
	box->addWidget( view );
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

void QLineNumberTextEdit::textChanged( int pos, int removed, int added )
{
	Q_UNUSED( pos );

	if ( removed == 0 && added == 0 )
		return;

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


