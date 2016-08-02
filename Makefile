BIN=maglua

all: ${BIN}

${BIN}:
	make -C src
	rm -f maglua
	ln -s src/${BIN} .

install: $(BIN)
	make -C src install

pack: clean
	cd .. && tar -czf maglua-`date +"%Y-%m-%d-%H.%M.%S"`.tar.gz maglua
	echo "Created archive: `ls ../maglua*-*-*gz | tail -n 1`"

clean:
	make -C src clean
	rm -f *~
	rm -f ${BIN}
