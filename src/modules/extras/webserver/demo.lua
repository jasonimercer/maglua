-- This is a WebServer demo script
--
-- It will start a web server on port 8888 so navigating to
-- http://localhost:8888 interacts with this script.
-- This web server will serve 3 pages, one unnamed, one
-- named names.php and another named action_page.php.
-- The main page in this example is a static page while
-- the other pages have content populated by MagLua functions.

ws = WebServer.new()
ws:setPort(8888)

-- this will be the "database" of names entered by the clients
local names = {}

local HTTP_OK = 200

ws:addPage("/", 
       [[
          <html>
           <body>
            <form action="action_page.php">
             First name:<br>
             <input type="text" name="firstname" value="">
              <br>
              Last name:<br>
             <input type="text" name="lastname" value="">
              <br><br>
             <input type="submit" value="Submit">
            </form>
            current <A HREF="names.php">names</A><br>
           </body>
          </html>
]])

ws:addPage("/action_page.php", 
           function(query)
               if query.firstname and query.lastname then
                   local name = query.firstname  .. " " .. query.lastname
                   table.insert(names, name)
               end

               return HTTP_OK, [[
                           <HTML>
                            <HEAD>
                             <meta http-equiv="refresh" content="2;url=/" />
                            </HEAD>
                            <BODY>
                             Name added, returning to main page...
                            </BODY>
                           </HTML>
                   ]]
           end )

ws:addPage("/names.php", 
           function(query)
               return HTTP_OK, 
               "<HTML><BODY>Names:<br>" .. 
                   table.concat(names, "<br>\n") .. 
               "</BODY></HTML>"
           end
       )

-- start the web server
print("Starting the webserver")
ws:start()

