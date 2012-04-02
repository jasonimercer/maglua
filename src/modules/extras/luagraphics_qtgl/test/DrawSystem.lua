local function arrow(x,y)
	local global_translate = Translate.new()
	local local_phi = Rotate.new()
	local local_theta = Rotate.new()
	local head_translate = Translate.new()
	local scale = Scale.new(0.5,0.5,0.5)
	local head1 = Tube.new()
	local head2 = Tube.new()
	local head3 = Tube.new()
	local body = Tube.new()
	local color = Color.new(1,0,0,1)
	local white = Color.new(1,1,1,1)
	
	local r = 0.5
	local p1, p2, p3 = 0.2, 0.45, 0.8
	
	head1:setPosition(1, 0, 0, 0.0)
	head1:setPosition(2, 0, 0, p1)
	head1:setRadius(1, r)
	head1:setRadius(2, r * (1 - p1/p3))
	head1:setColor(color)
	
	head2:setPosition(1, 0, 0, p1)
	head2:setPosition(2, 0, 0, p2)
	head2:setRadius(1, r * (1 - p1/p3))
	head2:setRadius(2, r * (1 - p2/p3))
	head2:setColor(white)

	head3:setPosition(1, 0, 0, p2)
	head3:setPosition(2, 0, 0, p3)
	head3:setRadius(1, r * (1 - p2/p3))
	head3:setRadius(2, r * (1 - p3/p3))
	head3:setColor(color)
	
	body:setPosition(1, 0, 0, -1/2)
	body:setPosition(2, 0, 0,  1/2)
	body:setRadius(1, 0.25)
	body:setRadius(2, 0.25)
	body:setColor(color)
	
	head_translate:set(0,0,0.5)
	head_translate:add(head1,head2,head3)

	scale:add(head_translate, body)
	
	local_phi:add(scale)
	local_theta:add(local_phi)

	global_translate:add(local_theta)
	global_translate:set(x,y,0)
	
	return {global_translate, local_theta, local_phi, color}
end

function makeSysetm(ss)
	local ssg = {}
	local all_arrows = Translate.new()
	for i=1,n do
		ssg[i] = {}
		for j=1,n do
			
			local x = arrow(i, j)
			
			ssg[i][j] = {}
			ssg[i][j].translate = x[1]
			ssg[i][j].theta = x[2]
			ssg[i][j].phi = x[3]
			ssg[i][j].color = x[4]

			local function makeSetter()
				local y = x
				return function(t,p)
					y[3]:set(0,p * 180/math.pi,0)
					y[2]:set(0,0,t * 180/math.pi)
					y[4]:map(t,p)
				end
			end
			ssg[i][j].set = makeSetter()
			
			all_arrows:add(x[1])
		end
	end

	return all_arrows, ssg
end

