module.exports = (sequelize, DataTypes) => {
    return sequelize.define('CLIENT', {
      id: {
        type: DataTypes.INTEGER,
        primaryKey: true,
        autoIncrement: true
      },
      nom_client: {
        type: DataTypes.STRING,

      },
      
      prenom_client: {
        type: DataTypes.STRING,
      },
    
      
       phone:{
        type:DataTypes.INTEGER,
       }

    })
  }