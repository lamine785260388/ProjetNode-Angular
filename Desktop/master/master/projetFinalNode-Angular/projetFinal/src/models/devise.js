module.exports = (sequelize, DataTypes) => {
    return sequelize.define('DEVISE', {
      id: {
        type: DataTypes.INTEGER,
        primaryKey: true,
        autoIncrement: true
      },
      code_iso3:{
      type:DataTypes.STRING
      },
      nom_devise: {
        type: DataTypes.STRING,

      },
  
      
      symbole: {
        type: DataTypes.STRING,
      },
      
     
       

    })
  }