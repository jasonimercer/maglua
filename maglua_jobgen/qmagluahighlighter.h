#ifndef QMAGLUAHIGHLIGHTER_H
#define QMAGLUAHIGHLIGHTER_H

#include <QSyntaxHighlighter>

class QMagLuaHighlighter : public QSyntaxHighlighter
{
	Q_OBJECT
public:
	enum Construct {
		Entity,
		String,
		Keyword,
		Variable,
		Function,
		Comment,
		LastConstruct = Comment
	};

	QMagLuaHighlighter(QTextDocument *document=0);

	void setFormatFor(Construct construct,
					  const QTextCharFormat &format);
	QTextCharFormat formatFor(Construct construct) const
	{ return m_formats[construct]; }

	QTextCharFormat variableFormat;
	QTextCharFormat keywordFormat;
	QTextCharFormat functionFormat;
	QTextCharFormat stringFormat;
	QTextCharFormat commentFormat;

//public slots:
//	void currentStateChanged(LuaStateTree* state);


protected:
	enum
	{
		NormalState = -1,
		InsideString
	};

	void highlightBlock(const QString &text);

private:
	QTextCharFormat m_formats[LastConstruct + 1];

	QStringList keywords;
	QStringList functions;
	QStringList variables;

	struct HighlightingRule
	{
		QRegExp pattern;
		QTextCharFormat format;
	};
	QVector<HighlightingRule> highlightingRules;

#if 0
    Q_OBJECT
public:
	explicit QMagLuaHighlighter(QTextDocument *parent = 0);

protected:
	void highlightBlock(const QString &text);

private:
	struct HighlightingRule
	{
		QRegExp pattern;
		QTextCharFormat format;
	};
	QVector<HighlightingRule> highlightingRules;

	QRegExp commentStartExpression;
	QRegExp commentEndExpression;

	QTextCharFormat keywordFormat;
	QTextCharFormat assignmentFormat;
	QTextCharFormat singleLineCommentFormat;
	QTextCharFormat multiLineCommentFormat;
	QTextCharFormat quotationFormat;
	QTextCharFormat functionFormat;
#endif
};

#endif // QMAGLUAHIGHLIGHTER_H
