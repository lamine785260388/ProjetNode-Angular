
const { Client } = require('../../db/sequelize')

  
module.exports = (app) => {
  app.post('/api/InsertClient', (req, res) => {
  
  
      Client.findOne({where:{CNI:+req.body.CNI}})
      .then(agence => {
        if(agence==null){
          return Client.create({
            
            nom_client:req.body.nom_client,
            CNI:req.body.CNI,
            prenom_client:req.body.prenom_client,
            phone:req.body.phone
          })
          .then(result=>{
            const message = 'creation avec succés'
            const idem=result.id
            console.log(idem)
            res.json({data:result.CNI,idem})
          })
          .catch(error=>{
            console.log(error);
            res.json({data:error})
          })
        }
        const idem=agence.id
        console.log(idem)
        const message = 'La liste des Clients a bien été récupéréezffz.'
        res.json({ message, data: agence.CNI,idem })
      })
     .catch(error=>{
      const message="la liste des Clients n'est pas disponibles.Résseyer dans quelques instant!"
      res.status(500).json({message,data:error})
     })
    
   
  })
}