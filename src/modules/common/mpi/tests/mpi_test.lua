-- Example on splitting an MPI_Comm into sub-workgroups
-- This script splits the global group into 3 groups of 
-- sizes 4, 3 and 1.
rank = mpi.get_rank()
size = mpi.get_size()

if size ~= 8 then
	error("Please run test with n = 8")
end

new_group = {1,1,1,2,2,1,5,2}

split_comm = mpi.comm_split(new_group[rank])
split_rank = split_comm:get_rank()
split_size = split_comm:get_size()

-- custom pathways for subsets
if new_group[rank] == 1 then
	t = split_comm:bcast(1, split_comm:gather(1, rank))
	t = table.concat(t, ",")
end
if new_group[rank] == 2 then
	split_comm:barrier()
	t = "Group 2"
end

print(rank, size, split_rank, split_size, new_group[rank],t)
-- Sample output
-- mpirun -n 8 maglua -q mpi_test.lua | sort -n
-- 1	8	1	4	1	1,2,3,6
-- 2	8	2	4	1	1,2,3,6
-- 3	8	3	4	1	1,2,3,6
-- 4	8	1	3	2	Group 2
-- 5	8	2	3	2	Group 2
-- 6	8	4	4	1	1,2,3,6
-- 7	8	1	1	3	nil
-- 8	8	3	3	2	Group 2
