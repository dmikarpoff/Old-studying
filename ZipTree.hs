import Prelude

data Tree a = Node a [Tree a] | Leaf

data ZipTree a = ZipTree { dir :: [Int]
                         , value :: [a]
			 , subTree :: [[Tree a]]
                         , finalTree :: Tree a
                         }

toChild :: ZipTree a -> Int -> ZipTree a
toChild z pos = case (finalTree z) of
     Node a lx -> ZipTree { dir = pos : (dir z)
			   , value = a : (value z)
			   , subTree = (lx `del` pos) : (subTree z)
                     	   , finalTree = lx !! pos
			   }
     Leaf -> z

del :: [a] -> Int -> [a]
del [] pos = []
del (x: lx) pos | pos >0   = x: (del lx (pos-1))
                | pos == 0 = lx   

ins :: [a] -> Int -> a -> [a]
ins [] pos val = [val]
ins (x:lx) pos val | pos > 0  = x: (ins lx (pos-1) val)
		   | pos == 0 = val : x : lx

{-toRight :: ZipTree a -> ZipTree a
toRight z = case (finalTree z) of
     Node a l r -> ZipTree { dir = R : (dir z)
			   , value = a : (value z)
			   , subTree = l : (subTree z)
			   , finalTree = r
			   }
     Leaf -> z
-}

toParent :: ZipTree a -> ZipTree a
toParent z = case (dir z) of
     [] -> z
     d:ld -> ZipTree { dir = ld
		     , value = tail $ value z
		     , subTree = tail $ subTree z
		     , finalTree = Node (head $ value z) (ins (head $ subTree z) d (finalTree z))
		     }

initZip :: Tree a -> ZipTree a
initZip tree = ZipTree{ dir = []
		      , value = []
		      , subTree = []
		      , finalTree = tree
		      }

delRoot :: Tree a -> Tree a
delRoot Leaf = Leaf
delRoot (Node a l) = chooseRoot l (findNode l 0 (length l))
   where findNode (x:lx) i len | i == len = len
			       | i < len  = case x of
				       Leaf        -> findNode (x:lx) (i+1) len    
				       Node a list -> i

chooseRoot :: [Tree a] -> Int -> Tree a
chooseRoot lt pos | pos >= (length lt) = Leaf
                  | otherwise	       = case lt !! pos of
					Leaf        -> Leaf
					Node a list -> Node a ((take (pos+1) lt) ++ list ++ (drop (pos+1) lt))

delVertex :: ZipTree a -> ZipTree a
delVertex z = case finalTree z of
             Leaf -> z
	     Node a lx -> ZipTree{  dir = dir z
		     		 ,  value = value z
		     		 ,  subTree = subTree z
                     		 ,  finalTree = delRoot $ finalTree z
		     		 }

makeChildList :: Tree a -> Int -> Int -> [Tree a]
makeChildList tree amount pos | amount > 0 = if (pos == 0)
						then tree: ( makeChildList tree (amount-1) (pos-1))
						else Leaf: ( makeChildList tree (amount-1) (pos-1))
			      | otherwise  = []  

insVertex :: ZipTree a -> Int -> Int -> a -> ZipTree a
insVertex z pos amount val = ZipTree { dir = dir z
				       , value = value z
                                       , subTree = subTree z
                                       , finalTree = Node val $ makeChildList (finalTree z) amount pos 	
				       }


