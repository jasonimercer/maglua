#ifndef QLHighlighter
#define QLHighlighter

#include <QSyntaxHighlighter>
#include <QRegExp>

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

class QLuaHilighter : public QSyntaxHighlighter
{
	Q_OBJECT

public:
	QLuaHilighter(QTextDocument *document=0);

protected:
	void highlightBlock(const QString &text);

private:
	struct HighlightingRule
	{
		QRegExp pattern;
		QTextCharFormat format;
	};
	QVector<HighlightingRule> highlightingRules;


	QStringList keywords;
	QStringList functions;
	QStringList variables;

	QRegExp commentStartExpression;
	QRegExp commentEndExpression;

	QTextCharFormat defaultFormat;
	QTextCharFormat keywordFormat;
	QTextCharFormat numberFormat;
	QTextCharFormat functionFormat;
	QTextCharFormat specialFormat;
	QTextCharFormat classFormat;
	QTextCharFormat singleLineCommentFormat;
	QTextCharFormat multiLineCommentFormat;
	QTextCharFormat quotationFormat;
};

#endif


