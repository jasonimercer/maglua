# Copyright (C) 2008-2011 Jason Mercer.  All rights reserved.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

LIBBASE=.
LUALIBS=$(LIBBASE)/libLuaClient/libLuaClient.a \
        $(LIBBASE)/libLuaServer/libLuaServer.a \
	$(LIBBASE)/libLuaSqlite/libLuaSqlite.a \
	$(LIBBASE)/libLuaMigrate/libLuaMigrate.a \
	$(LIBBASE)/libLuaMysql/libLuaMysql.a

SUBDIRS=libLuaClient libLuaServer libLuaSqlite libLuaMigrate sampleSQLiteServer libLuaMysql

all: $(LUALIBS)

#to make each library, run make in their directory $(@D)
%.a:
	make -C $(@D)

test: all
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir test; \
	done
	make -C sampleSQLiteServer

#clean here means clean all libs
clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
	rm -f *~

distclean: clean
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir distclean; \
	done
