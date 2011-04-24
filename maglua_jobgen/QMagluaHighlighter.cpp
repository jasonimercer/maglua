#include "QMagluaHighlighter.h"

#if 0
QMagLuaHighlighter::QMagLuaHighlighter(QTextDocument *parent)
	: QSyntaxHighlighter(parent)
{
	HighlightingRule rule;

	keywordFormat.setForeground(Qt::black);
	keywordFormat.setFontWeight(QFont::Bold);
	QStringList keywordPatterns;
	keywordPatterns << "function" << "for" << "do"
			<< "while" << "repeat" << "if"
			<< "end" << "return" << "until";
	foreach (const QString &pattern, keywordPatterns)
	{
		rule.pattern = QRegExp(QString("\\b%1\\b").arg(pattern));
		rule.format = keywordFormat;
		highlightingRules.append(rule);
	}

	assignmentFormat.setForeground(Qt::darkGreen);
	rule.pattern = QRegExp("\\b=\\b");
	rule.format = assignmentFormat;
	highlightingRules.append(rule);


	//	 classFormat.setFontWeight(QFont::Bold);
	//	 classFormat.setForeground(Qt::darkMagenta);
	//	 rule.pattern = QRegExp("\\bQ[A-Za-z]+\\b");
	//	 rule.format = classFormat;
	//	 highlightingRules.append(rule);

	singleLineCommentFormat.setForeground(Qt::darkGray);
	rule.pattern = QRegExp("\\s*\\-\\-[^\n]*");
	rule.format = singleLineCommentFormat;
	highlightingRules.append(rule);

	multiLineCommentFormat.setForeground(Qt::red);

	quotationFormat.setForeground(Qt::darkGreen);
	rule.pattern = QRegExp("\".*\"");
	rule.format = quotationFormat;
	highlightingRules.append(rule);

	functionFormat.setFontItalic(true);
	functionFormat.setForeground(Qt::blue);
	rule.pattern = QRegExp("\\b[A-Za-z0-9_]+(?=\\()");
	rule.format = functionFormat;
	highlightingRules.append(rule);

	commentStartExpression = QRegExp("--[[");
	commentEndExpression = QRegExp("--]]");
}

void QMagLuaHighlighter::highlightBlock(const QString &text)
{
	foreach (const HighlightingRule &rule, highlightingRules) {
		QRegExp expression(rule.pattern);
		int index = expression.indexIn(text);
		while (index >= 0) {
			int length = expression.matchedLength();
			setFormat(index, length, rule.format);
			index = expression.indexIn(text, index + length);
		}
	}
	setCurrentBlockState(0);

	int startIndex = 0;
	if (previousBlockState() != 1)
		startIndex = commentStartExpression.indexIn(text);

	while (startIndex >= 0) {
		int endIndex = commentEndExpression.indexIn(text, startIndex);
		int commentLength;
		if (endIndex == -1) {
			setCurrentBlockState(1);
			commentLength = text.length() - startIndex;
		} else {
			commentLength = endIndex - startIndex
							+ commentEndExpression.matchedLength();
		}
		setFormat(startIndex, commentLength, multiLineCommentFormat);
		startIndex = commentStartExpression.indexIn(text, startIndex + commentLength);
	}
}
#endif

#include <iostream>
using namespace std;

QMagLuaHighlighter::QMagLuaHighlighter(QTextDocument *parent)
	: QSyntaxHighlighter(parent)
{
	QTextCharFormat entityFormat;
	entityFormat.setForeground(QColor(0, 128, 0));
	entityFormat.setFontWeight(QFont::Bold);

	stringFormat.setForeground(QColor(255, 0, 0));
	{
		QFont font;
		font.setFamily("Courier");
		font.setFixedPitch(true);
		font.setItalic(false);
		font.setBold(false);
		stringFormat.setFont(font);
	}

	commentFormat.setForeground(QColor(128, 128, 128));
	{
		QFont font;
		font.setFamily("Courier");
		font.setFixedPitch(true);
		font.setItalic(true);
		font.setBold(false);
		commentFormat.setFont(font);
	}


	keywordFormat.setForeground(QColor(0,0,0));
	{
		QFont font;
		font.setFamily("Courier");
		font.setFixedPitch(true);
		font.setItalic(false);
		font.setBold(true);
		keywordFormat.setFont(font);
	}

	functionFormat.setForeground(QColor(0,0,128));
	{
		QFont font;
		font.setFamily("Courier");
		font.setFixedPitch(true);
		font.setItalic(false);
		font.setBold(true);
		functionFormat.setFont(font);
	}
	variableFormat.setForeground(QColor(0,128,0));
	{
		QFont font;
		font.setFamily("Courier");
		font.setFixedPitch(true);
		font.setItalic(false);
		font.setBold(false);
		variableFormat.setFont(font);
	}

	setFormatFor(Entity, entityFormat);
	setFormatFor(String, stringFormat);
	setFormatFor(Comment, commentFormat);
	setFormatFor(Keyword, keywordFormat);
	setFormatFor(Function, functionFormat);
	setFormatFor(Variable, variableFormat);


	keywords << "and";
	keywords << "break";
	keywords << "do";
	keywords << "else";
	keywords << "elseif";
	keywords << "end";
	keywords << "false";
	keywords << "for";
	keywords << "function";
	keywords << "if";
	keywords << "in";
	keywords << "local";
	keywords << "nil";
	keywords << "not";
	keywords << "or";
	keywords << "repeat";
	keywords << "return";
	keywords << "then";
	keywords << "true";
	keywords << "until";
	keywords << "while";

	//	foreach (const QString &pattern, keywords)
	//	{
	//		HighlightingRule rule;
	//		rule.pattern = QRegExp(QString("\\b%1\\b").arg(pattern));
	//		rule.format = keywordFormat;
	//		highlightingRules.append(rule);
	//	}
}

void QMagLuaHighlighter::setFormatFor(Construct construct,
									  const QTextCharFormat &format)
{
	m_formats[construct] = format;
	rehighlight();
}

#if 0
void QMagLuaHighlighter::currentStateChanged(LuaStateTree* state)
{
	vector<string> vexclude;

	vexclude.push_back(string("_G"));
	vexclude.push_back(string("__index"));
	vexclude.push_back(string("package"));

	vector<string> vfunctions;
	state->typeNames("function", vfunctions, vexclude);

	functions.clear();
	for(unsigned int i=0; i<vfunctions.size(); i++)
		functions << vfunctions[i].c_str();

	vector<string> vvars;
	state->typeNames("number",  vvars, vexclude);
	state->typeNames("boolean", vvars, vexclude);
	state->typeNames("string",  vvars, vexclude);
	state->typeNames("thread",  vvars, vexclude);
	state->typeNames("object",  vvars, vexclude);
	state->typeNames("table",   vvars, vexclude);

	variables.clear();
	for(unsigned int i=0; i<vvars.size(); i++)
		variables << vvars[i].c_str();

	rehighlight();
}
#endif


static bool ok_prefix(int i, QString text)
{
	if(i == 0)
		return true;

	QString b = text.mid(i-1, 1); //before

	if(b == " " || b == "," || b == "(" || b == ")" || b == "\t" || b == "," || b == "{" || b == "}" || b == "=" || b == "[" || b == "]" || b == "+" || b == "-" || b == "*" || b == "/")
		return true;

	return false;
}
static bool ok_postfix(int i, QString text)
{
	if(i == text.length()-1)
		return true;

	QString b = text.mid(i+1, 1);

	if(b == " " || b == "," || b == "(" || b == ")" || b == "\t" || b == "," || b == "{" || b == "}" || b == "=" || b == "[" || b == "]" || b == "+" || b == "-" || b == "*" || b == "/")
		return true;

	return false;
}

static bool escaped(int i, QString text)
{
	if(i == 0)
		return false;
	if(text.mid(i-1, 1) == "\\")
		return true;
	return false;
}

void QMagLuaHighlighter::highlightBlock(const QString &text)
{
	//	QLuaTextBlockUserData* ud = (QLuaTextBlockUserData*)currentBlockUserData();

	//	if(!ud || ud->type != QLuaTextBlockUserData::input)
	//		return;

	int state = previousBlockState();
	int start = 0;
	bool gotword;

	for(int i=0; i<text.length(); ++i)
	{
		switch(state)
		{
		default:
		case NormalState:
			gotword = false;
			for(int j=0; j<keywords.size() && !gotword; j++)
			{
				cout << keywords.at(j).toStdString() << endl;
				if(text.mid(i, keywords.at(j).length()) == keywords.at(j))
				{
					//if(ok_prefix(i, text) && ok_postfix(i + keywords.at(j).length()-1, text))
					{
						gotword = true;
						setFormat(i, keywords.at(j).length(), m_formats[Keyword]);
						i += keywords.at(j).length();
					}
				}
			}

			if(!gotword)
				for(int j=0; j<functions.size() && !gotword; j++)
				{
				if(text.mid(i, functions.at(j).length()) == functions.at(j))
				{
					if(ok_prefix(i, text) && ok_postfix(i + functions.at(j).length()-1, text))
					{
						gotword = true;
						setFormat(i, functions.at(j).length(), m_formats[Function]);
						i += functions.at(j).length();
					}
				}
			}

			if(!gotword)
			{
				for(int j=0; j<variables.size() && !gotword; j++)
				{
					if(text.mid(i, variables.at(j).length()) == variables.at(j))
					{
						if(ok_prefix(i, text) && ok_postfix(i + variables.at(j).length()-1, text))
						{
							gotword = true;
							setFormat(i, variables.at(j).length(), m_formats[Variable]);
							i += variables.at(j).length();
						}
					}
				}
			}

			if(!gotword)
			{
				if(text.mid(i, 2) == "--")
				{
					setFormat(i, text.length() - i, m_formats[Comment]);
					i = text.length();
					break;
				}
				else if (text.mid(i, 1) == "\"" && !escaped(i, text))
				{
					start = i;
					state = InsideString;
					setFormat(start, i - start + 1, m_formats[String]);
				}
			}
			break;

			case InsideString:
			if (text.mid(i, 1) == "\"" && !escaped(i, text))
				state = NormalState;
			setFormat(start, i - start + 1, m_formats[String]);
			break;

		}
	}

	if(state == InsideString)
		state = NormalState; //strings don't span lines

	setCurrentBlockState(state);
}

