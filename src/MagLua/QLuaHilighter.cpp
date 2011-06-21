#include "QLuaHilighter.h"

#include <iostream>
using namespace std;

QLuaHilighter::QLuaHilighter(QTextDocument *parent)
	: QSyntaxHighlighter(parent)
{
	HighlightingRule rule;

	// default formatting
	defaultFormat.setForeground(QColor(22, 22, 255));
	rule.pattern = QRegExp("\\b.+\\b");
	rule.format = defaultFormat;
	highlightingRules.append(rule);

	numberFormat.setForeground(QColor(176, 128, 0));
	rule.format = numberFormat;
	rule.pattern = QRegExp("\\-?\\d+\\.\\d+");
	highlightingRules.append(rule);
	rule.pattern = QRegExp("\\-?\\d+");
	highlightingRules.append(rule);
	rule.pattern = QRegExp("\\-?\\d+\\.?\\d*[eE]\\-?\\d+");
	highlightingRules.append(rule);


	keywords << "\\band";
	keywords << "\\bbreak\\b";
	keywords << "\\bdo\\b";
	keywords << "\\belse\\b";
	keywords << "\\bend\\b";
	keywords << "\\bfalse\\b";
	keywords << "\\bfor\\b";
	keywords << "\\bfunction\\b";
	keywords << "\\bif\\b";
	keywords << "\\bin\\b";
	keywords << "\\blocal\\b";
	keywords << "\\bnil\\b";
	keywords << "\\bnot\\b";
	keywords << "\\bo\\br";
	keywords << "\\brepeat\\b";
	keywords << "\\blocal\\b";
	keywords << "\\breturn\\b";
	keywords << "\\bthen\\b";
	keywords << "\\btrue\\b";
	keywords << "\\buntil\\b";
	keywords << "\\bwhile\\b";


	functions << "\\bstring\\.sub\\b";
	functions << "\\bstring\\.upper\\b";
	functions << "\\bstring\\.len\\b";
	functions << "\\bstring\\.gfind\\b";
	functions << "\\bstring\\.rep\\b";
	functions << "\\bstring\\.find\\b";
	functions << "\\bstring\\.match\\b";
	functions << "\\bstring\\.char\\b";
	functions << "\\bstring\\.dump\\b";
	functions << "\\bstring\\.gmatch\\b";
	functions << "\\bstring\\.reverse\\b";
	functions << "\\bstring\\.byte\\b";
	functions << "\\bstring\\.format\\b";
	functions << "\\bstring\\.gsub\\b";
	functions << "\\bstring\\.lower\\b";
	functions << "\\bxpcall\\b";
	functions << "\\bpackage\\.loadlib\\b";
	functions << "\\bpackage\\.seeall\\b";
	functions << "\\btostring\\b";
	functions << "\\bprint\\b";
	functions << "\\bInterpolate2D\\.metatable\\b";
	functions << "\\bInterpolate2D\\.help\\b";
	functions << "\\bInterpolate2D\\.new\\b";
	functions << "\\bos\\.exit\\b";
	functions << "\\bos\\.setlocale\\b";
	functions << "\\bos\\.date\\b";
	functions << "\\bos\\.getenv\\b";
	functions << "\\bos\\.difftime\\b";
	functions << "\\bos\\.remove\\b";
	functions << "\\bos\\.time\\b";
	functions << "\\bos\\.clock\\b";
	functions << "\\bos\\.tmpname\\b";
	functions << "\\bos\\.rename\\b";
	functions << "\\bos\\.execute\\b";
	functions << "\\bunpack\\b";
	functions << "\\bClient\\.new\\b";
	functions << "\\brequire\\b";
	functions << "\\bgetfenv\\b";
	functions << "\\bLLG\\.metatable\\b";
	functions << "\\bLLG\\.help\\b";
	functions << "\\bLLG\\.new\\b";
	functions << "\\bsetmetatable\\b";
	functions << "\\bnext\\b";
	functions << "\\bassert\\b";
	functions << "\\btonumber\\b";
	functions << "\\binfo\\b";
	functions << "\\bio\\.lines\\b";
	functions << "\\bio\\.write\\b";
	functions << "\\bio\\.close\\b";
	functions << "\\bio\\.flush\\b";
	functions << "\\bio\\.open\\b";
	functions << "\\bio\\.output\\b";
	functions << "\\bio\\.type\\b";
	functions << "\\bio\\.read\\b";
	functions << "\\bio\\.input\\b";
	functions << "\\bio\\.popen\\b";
	functions << "\\bio\\.tmpfile\\b";
	functions << "\\brawequal\\b";
	functions << "\\bnewproxy\\b";
	functions << "\\bcollectgarbage\\b";
	functions << "\\bgetmetatable\\b";
	functions << "\\bSQL\\.open\\b";
	functions << "\\bSQL\\.new\\b";
	functions << "\\bmodule\\b";
	functions << "\\bInterpolate\\.metatable\\b";
	functions << "\\bInterpolate\\.help\\b";
	functions << "\\bInterpolate\\.new\\b";
	functions << "\\bSpinSystem\\.metatable\\b";
	functions << "\\bSpinSystem\\.help\\b";
	functions << "\\bSpinSystem\\.new\\b";
	functions << "\\bipairs\\b";
	functions << "\\bgcinfo\\b";
	functions << "\\brawset\\b";
	functions << "\\bThermal\\.metatable\\b";
	functions << "\\bThermal\\.help\\b";
	functions << "\\bThermal\\.new\\b";
	functions << "\\bAnisotropy\\.metatable\\b";
	functions << "\\bAnisotropy\\.help\\b";
	functions << "\\bAnisotropy\\.new\\b";
	functions << "\\bRandom\\.metatable\\b";
	functions << "\\bRandom\\.help\\b";
	functions << "\\bRandom\\.new\\b";
	functions << "\\bMagnetostatic\\.metatable\\b";
	functions << "\\bMagnetostatic\\.help\\b";
	functions << "\\bMagnetostatic\\.new\\b";
	functions << "\\bExchange\\.metatable\\b";
	functions << "\\bExchange\\.help\\b";
	functions << "\\bExchange\\.new\\b";
	functions << "\\bmath\\.log\\b";
	functions << "\\bmath\\.max\\b";
	functions << "\\bmath\\.acos\\b";
	functions << "\\bmath\\.ldexp\\b";
	functions << "\\bmath\\.cos\\b";
	functions << "\\bmath\\.tanh\\b";
	functions << "\\bmath\\.pow\\b";
	functions << "\\bmath\\.deg\\b";
	functions << "\\bmath\\.tan\\b";
	functions << "\\bmath\\.cosh\\b";
	functions << "\\bmath\\.sinh\\b";
	functions << "\\bmath\\.random\\b";
	functions << "\\bmath\\.randomseed\\b";
	functions << "\\bmath\\.frexp\\b";
	functions << "\\bmath\\.ceil\\b";
	functions << "\\bmath\\.floor\\b";
	functions << "\\bmath\\.rad\\b";
	functions << "\\bmath\\.abs\\b";
	functions << "\\bmath\\.sqrt\\b";
	functions << "\\bmath\\.modf\\b";
	functions << "\\bmath\\.asin\\b";
	functions << "\\bmath\\.min\\b";
	functions << "\\bmath\\.mod\\b";
	functions << "\\bmath\\.fmod\\b";
	functions << "\\bmath\\.log10\\b";
	functions << "\\bmath\\.atan2\\b";
	functions << "\\bmath\\.exp\\b";
	functions << "\\bmath\\.sin\\b";
	functions << "\\bmath\\.atan\\b";
	functions << "\\bdebug\\.getupvalue\\b";
	functions << "\\bdebug\\.debug\\b";
	functions << "\\bdebug\\.sethook\\b";
	functions << "\\bdebug\\.getmetatable\\b";
	functions << "\\bdebug\\.gethook\\b";
	functions << "\\bdebug\\.setmetatable\\b";
	functions << "\\bdebug\\.setlocal\\b";
	functions << "\\bdebug\\.traceback\\b";
	functions << "\\bdebug\\.setfenv\\b";
	functions << "\\bdebug\\.getinfo\\b";
	functions << "\\bdebug\\.setupvalue\\b";
	functions << "\\bdebug\\.getlocal\\b";
	functions << "\\bdebug\\.getregistry\\b";
	functions << "\\bdebug\\.getfenv\\b";
	functions << "\\bpcall\\b";
	functions << "\\btable\\.setn\\b";
	functions << "\\btable\\.insert\\b";
	functions << "\\btable\\.getn\\b";
	functions << "\\btable\\.foreachi\\b";
	functions << "\\btable\\.maxn\\b";
	functions << "\\btable\\.foreach\\b";
	functions << "\\btable\\.concat\\b";
	functions << "\\btable\\.sort\\b";
	functions << "\\btable\\.remove\\b";
	functions << "\\bAppliedField\\.metatable\\b";
	functions << "\\bAppliedField\\.help\\b";
	functions << "\\bAppliedField\\.new\\b";
	functions << "\\btype\\b";
	functions << "\\bcoroutine\\.resume\\b";
	functions << "\\bcoroutine\\.yield\\b";
	functions << "\\bcoroutine\\.status\\b";
	functions << "\\bcoroutine\\.wrap\\b";
	functions << "\\bcoroutine\\.create\\b";
	functions << "\\bcoroutine\\.running\\b";
	functions << "\\bselect\\b";
	functions << "\\bDisorderedDipole\\.metatable\\b";
	functions << "\\bDisorderedDipole\\.help\\b";
	functions << "\\bDisorderedDipole\\.new\\b";
	functions << "\\bpairs\\b";
	functions << "\\brawget\\b";
	functions << "\\bloadstring\\b";
	functions << "\\bDipole\\.metatable\\b";
	functions << "\\bDipole\\.help\\b";
	functions << "\\bDipole\\.new\\b";
	functions << "\\bdofile\\b";
	functions << "\\bsetfenv\\b";
	functions << "\\bload\\b";
	functions << "\\berror\\b";
	functions << "\\bloadfile\\b";

	variables << "==" << "~=" << "=" << "\\{" << "\\}" << "\\[" << "\\]";
	variables << "<=" << ">=" << "\\(" << "\\)";
	variables << "<" << ">" << "\\+" << "\\*" << "\\-" << "\\/";

	specialFormat.setForeground(QColor(0, 110, 40));
	foreach (const QString &pattern, variables)
	{
		rule.pattern = QRegExp(pattern);
		rule.format = specialFormat;
		highlightingRules.append(rule);
	}


	keywordFormat.setForeground(QColor(161,161,0));
	foreach (const QString &pattern, keywords) {
		rule.pattern = QRegExp(pattern);
		rule.format = keywordFormat;
		highlightingRules.append(rule);
	}


	//functionFormat.setForeground(QColor(100, 74, 155));
	functionFormat.setForeground(QColor(0, 0, 155));
	foreach (const QString &pattern, functions)
	{
		rule.pattern = QRegExp(pattern);
		rule.format = functionFormat;
		highlightingRules.append(rule);
	}




	singleLineCommentFormat.setForeground(Qt::darkGray);
	rule.pattern = QRegExp("--[^\n]*");
	rule.format = singleLineCommentFormat;
	highlightingRules.append(rule);

	multiLineCommentFormat.setForeground(Qt::red);

	//quotationFormat.setForeground(Qt::darkRed);
	quotationFormat.setForeground(Qt::red);
	rule.pattern = QRegExp("\".*\"");
	rule.format = quotationFormat;
	highlightingRules.append(rule);

//	functionFormat.setFontItalic(true);
//	functionFormat.setForeground(Qt::blue);
//	rule.pattern = QRegExp("\\b[A-Za-z0-9_]+(?=\\()");
//	rule.format = functionFormat;
//	highlightingRules.append(rule);



	commentStartExpression = QRegExp("\\-\\-\\[");
	commentEndExpression = QRegExp("\\-\\-\\]");
}

void QLuaHilighter::highlightBlock(const QString &text)
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



#if 0

QLuaHilighter::QLuaHilighter(QTextDocument *document)
	: QSyntaxHighlighter(document)
{
	QTextCharFormat entityFormat;
	entityFormat.setForeground(QColor(0, 128, 0));
	entityFormat.setFontWeight(QFont::Bold);
	setFormatFor(Entity, entityFormat);
 
	stringFormat.setForeground(QColor(255, 0, 0));
//	{
//		QFont font;
//		//font.setFamily("Mono");
//		font.setFixedPitch(true);
//		font.setItalic(false);
//		font.setBold(false);
//		stringFormat.setFont(font);
//	}
	setFormatFor(String, stringFormat);

	commentFormat.setForeground(QColor(128, 128, 128));
//	{
//		QFont font;
//		//font.setFamily("Mono");
//		font.setFixedPitch(true);
//		font.setItalic(true);
//		font.setBold(false);
//		commentFormat.setFont(font);
//	}
	setFormatFor(Comment, commentFormat);


	keywordFormat.setForeground(QColor(0,0,0));
	keywordFormat.setFontWeight(QFont::Bold);
//	{
//		QFont font;
//		//font.setFamily("Mono");
//		font.setFixedPitch(true);
//		font.setItalic(false);
//		font.setBold(true);
//		keywordFormat.setFont(font);
//	}

	functionFormat.setForeground(QColor(0,0,128));
//	{
//		QFont font;
//		//font.setFamily("Mono");
//		font.setFixedPitch(true);
//		font.setItalic(false);
//		font.setBold(true);
//		functionFormat.setFont(font);
//	}
	variableFormat.setForeground(QColor(0,128,0));
//	{
//		QFont font;
//		//font.setFamily("Mono");
//		font.setFixedPitch(true);
//		font.setItalic(false);
//		font.setBold(false);
//		variableFormat.setFont(font);
//	}

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

// --lua code to get functions
//	for k,v in pairs(_G) do
//		if type(v) == "function" then
//			print("functions << \"" .. k .. "\";")
//		end

//		if type(v) == "table" and k ~= "_G" then
//			for a,b in pairs(v) do
//				if type(b) == "function" then
//					print("functions << \"" .. k .. "." .. a .. "\";")
//				end
//			end
//		end
//	end

	functions << "string.sub";
	functions << "string.upper";
	functions << "string.len";
	functions << "string.gfind";
	functions << "string.rep";
	functions << "string.find";
	functions << "string.match";
	functions << "string.char";
	functions << "string.dump";
	functions << "string.gmatch";
	functions << "string.reverse";
	functions << "string.byte";
	functions << "string.format";
	functions << "string.gsub";
	functions << "string.lower";
	functions << "xpcall";
	functions << "package.loadlib";
	functions << "package.seeall";
	functions << "tostring";
	functions << "print";
	functions << "os.exit";
	functions << "os.setlocale";
	functions << "os.date";
	functions << "os.getenv";
	functions << "os.difftime";
	functions << "os.remove";
	functions << "os.time";
	functions << "os.clock";
	functions << "os.tmpname";
	functions << "os.rename";
	functions << "os.execute";
	functions << "unpack";
	functions << "require";
	functions << "getfenv";
	functions << "setmetatable";
	functions << "next";
	functions << "assert";
	functions << "tonumber";
	functions << "io.lines";
	functions << "io.write";
	functions << "io.close";
	functions << "io.flush";
	functions << "io.open";
	functions << "io.output";
	functions << "io.type";
	functions << "io.read";
	functions << "io.input";
	functions << "io.popen";
	functions << "io.tmpfile";
	functions << "rawequal";
	functions << "collectgarbage";
	functions << "getmetatable";
	functions << "module";
	functions << "rawset";
	functions << "math.log";
	functions << "math.max";
	functions << "math.acos";
	functions << "math.ldexp";
	functions << "math.cos";
	functions << "math.tanh";
	functions << "math.pow";
	functions << "math.deg";
	functions << "math.tan";
	functions << "math.cosh";
	functions << "math.sinh";
	functions << "math.random";
	functions << "math.randomseed";
	functions << "math.frexp";
	functions << "math.ceil";
	functions << "math.floor";
	functions << "math.rad";
	functions << "math.abs";
	functions << "math.sqrt";
	functions << "math.modf";
	functions << "math.asin";
	functions << "math.min";
	functions << "math.mod";
	functions << "math.fmod";
	functions << "math.log10";
	functions << "math.atan2";
	functions << "math.exp";
	functions << "math.sin";
	functions << "math.atan";
	functions << "debug.getupvalue";
	functions << "debug.debug";
	functions << "debug.sethook";
	functions << "debug.getmetatable";
	functions << "debug.gethook";
	functions << "debug.setmetatable";
	functions << "debug.setlocal";
	functions << "debug.traceback";
	functions << "debug.setfenv";
	functions << "debug.getinfo";
	functions << "debug.setupvalue";
	functions << "debug.getlocal";
	functions << "debug.getregistry";
	functions << "debug.getfenv";
	functions << "pcall";
	functions << "table.setn";
	functions << "table.insert";
	functions << "table.getn";
	functions << "table.foreachi";
	functions << "table.maxn";
	functions << "table.foreach";
	functions << "table.concat";
	functions << "table.sort";
	functions << "table.remove";
	functions << "newproxy";
	functions << "type";
	functions << "coroutine.resume";
	functions << "coroutine.yield";
	functions << "coroutine.status";
	functions << "coroutine.wrap";
	functions << "coroutine.create";
	functions << "coroutine.running";
	functions << "select";
	functions << "gcinfo";
	functions << "pairs";
	functions << "rawget";
	functions << "loadstring";
	functions << "ipairs";
	functions << "dofile";
	functions << "setfenv";
	functions << "load";
	functions << "error";
	functions << "loadfile";

	variables << "==" << "~=" << "=";
	variables << "<=" << ">=";
	variables << "<" << ">";
	variables << "{" << "}";
	variables << "[" << "]";
}

void QLuaHilighter::setFormatFor(Construct construct, const QTextCharFormat &format)
{
		m_formats[construct] = format;
		rehighlight();
}


bool ok_prefix(int i, QString text)
{
	if(i == 0)
		return true;

	QString b = text.mid(i-1, 1); //before

	if(b == " " || b == "," || b == "(" || b == ")" || b == "\t" || b == "," || b == "{" || b == "}" || b == "=" || b == "[" || b == "]" || b == "+" || b == "-" || b == "*" || b == "/")
		return true;

	return false;
}
bool ok_postfix(int i, QString text)
{
	if(i == text.length()-1)
		return true;

	QString b = text.mid(i+1, 1);

	if(b == " " || b == "," || b == "(" || b == ")" || b == "\t" || b == "," || b == "{" || b == "}" || b == "=" || b == "[" || b == "]" || b == "+" || b == "-" || b == "*" || b == "/")
		return true;

	return false;
}

bool escaped(int i, QString text)
{
	if(i == 0)
		return false;
	if(text.mid(i-1, 1) == "\\")
		return true;
	return false;
}

void QLuaHilighter::highlightBlock(const QString &text)
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

				if(!gotword)
				for(int j=0; j<keywords.size() && !gotword; j++)
				{
					if(text.mid(i, keywords.at(j).length()) == keywords.at(j))
					{
						if(ok_prefix(i, text) && ok_postfix(i + keywords.at(j).length()-1, text))
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

#endif
